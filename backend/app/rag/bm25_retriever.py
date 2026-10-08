"""
BM25 Retriever for CA Mentor AI.

Purpose
-------
Retrieve educational chunks using lexical keyword matching.

Pipeline
--------
User Question
    ↓
Tokenization
    ↓
BM25 Search
    ↓
Ranked Candidate Chunks

BM25 is complementary to dense retrieval.

Dense retrieval:
    Understands semantic similarity.

BM25:
    Matches important words and phrases.

This module does NOT:
- generate answers
- call an LLM
- access Qdrant
- perform dense vector search
- perform reranking
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from app.core.logging import logger
from app.rag.dense_retriever import RetrievedChunk


class BM25RetrievedChunk(BaseModel):
    """
    One chunk returned by BM25 retrieval.
    """

    chunk: RetrievedChunk
    score: float


class BM25Retriever:
    """
    Lexical BM25 retriever over retrieved/indexed document chunks.

    Parameters
    ----------
    chunks:
        Collection of document chunks that BM25 will search.

    The chunk collection is kept in memory.

    This is intentional for the current CA Mentor AI
    implementation because the current corpus contains
    43 chunks.

    Later, when the corpus becomes large, this layer can
    be replaced with a persistent BM25 index without
    changing the retrieval interface.
    """

    def __init__(
        self,
        chunks: Iterable[RetrievedChunk],
    ) -> None:

        self.chunks = list(chunks)

        if not self.chunks:
            raise ValueError(
                "BM25Retriever requires at least one chunk."
            )

        documents = [
            self._tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self._bm25 = BM25Okapi(documents)

        logger.info(
            f"BM25 index initialized with "
            f"{len(self.chunks)} chunks."
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[BM25RetrievedChunk]:
        """
        Retrieve the most relevant chunks using BM25.
        """

        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        logger.info(
            f"BM25 retrieval started. "
            f"question={question!r}, top_k={top_k}"
        )

        query_tokens = self._tokenize(question)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[BM25RetrievedChunk] = []

        for index in ranked_indexes[:top_k]:

            score = float(scores[index])

            results.append(
                BM25RetrievedChunk(
                    chunk=self.chunks[index],
                    score=score,
                )
            )

        logger.info(
            f"BM25 retrieval completed. "
            f"Retrieved {len(results)} chunks."
        )

        return results

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        """
        Convert text into normalized BM25 tokens.

        Steps:
        1. Lowercase.
        2. Extract words/numbers.
        3. Preserve meaningful accounting terminology.
        """

        return re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )