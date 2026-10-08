"""
BGE Reranker for CA Mentor AI.

Purpose
-------
Rerank hybrid retrieval candidates using a cross-encoder.

Input:
    User question + retrieved chunks

Output:
    Same chunks ordered by relevance to the question.

Model:
    BAAI/bge-reranker-base

Pipeline:
    Dense Retrieval
          +
    BM25 Retrieval
          ↓
    Hybrid Retrieval
          ↓
    BGE Reranker
          ↓
    Best Evidence
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger
from app.rag.hybrid_retriever import HybridRetrievedChunk


class RerankedChunk(BaseModel):
    """
    A hybrid-retrieved chunk with its reranker relevance score.
    """

    chunk: HybridRetrievedChunk
    reranker_score: float
    rerank_rank: int


class BGEReranker:
    """
    Reranks retrieved chunks using BAAI/bge-reranker-base.

    The model is loaded lazily so importing this module does not
    immediately download or load the model.
    """

    DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"
    DEFAULT_TOP_K = 5

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        model: Any | None = None,
    ) -> None:

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.model_name = (
            model_name
            or settings.RERANKER_MODEL
            or self.DEFAULT_MODEL_NAME
        )

        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty.")

        self.top_k = top_k
        self._model = model

    @property
    def model(self) -> Any:
        """
        Lazily load the cross-encoder model.
        """

        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                f"Loading reranker model: {self.model_name}"
            )

            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
            )

            logger.info(
                f"Reranker model loaded successfully: "
                f"{self.model_name}"
            )

        return self._model

    def rerank(
        self,
        question: str,
        candidates: list[HybridRetrievedChunk],
        top_k: int | None = None,
    ) -> list[RerankedChunk]:
        """
        Rerank hybrid retrieval candidates.

        Args:
            question:
                User's question.

            candidates:
                Candidates produced by HybridRetriever.

            top_k:
                Number of final candidates to return.

        Returns:
            Reranked chunks ordered from most relevant
            to least relevant.
        """

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        if not candidates:
            logger.info(
                "Reranker received no candidates."
            )
            return []

        requested_top_k = (
            self.top_k if top_k is None else top_k
        )

        if requested_top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        logger.info(
            "Reranking started. "
            f"question={question!r}, "
            f"candidates={len(candidates)}, "
            f"top_k={requested_top_k}"
        )

        pairs = [
            [
                question,
                candidate.chunk.text,
            ]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        if len(scores) != len(candidates):
            raise RuntimeError(
                "Reranker returned a different number of "
                "scores than candidates."
            )

        scored_candidates: list[
            tuple[HybridRetrievedChunk, float]
        ] = []

        for candidate, score in zip(
            candidates,
            scores,
            strict=True,
        ):
            scored_candidates.append(
                (
                    candidate,
                    float(score),
                )
            )

        scored_candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[RerankedChunk] = []

        for rank, (candidate, score) in enumerate(
            scored_candidates[:requested_top_k],
            start=1,
        ):
            results.append(
                RerankedChunk(
                    chunk=candidate,
                    reranker_score=score,
                    rerank_rank=rank,
                )
            )

        logger.info(
            "Reranking completed. "
            f"Returned {len(results)} candidates."
        )

        return results