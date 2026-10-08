"""
Hybrid Retriever for CA Mentor AI.

Combines:
    1. Dense semantic retrieval
    2. BM25 lexical retrieval

Pipeline:

User Question
      |
      +--------------------+
      |                    |
      v                    v
Dense Retrieval        BM25 Retrieval
      |                    |
      +---------+----------+
                |
                v
        Candidate Fusion
                |
                v
        Hybrid Ranked Results

This module does NOT:
- generate answers
- call an LLM
- perform reranking
- modify Qdrant
- modify ingestion
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core.logging import logger
from app.rag.bm25_retriever import BM25Retriever
from app.rag.dense_retriever import DenseRetriever, RetrievedChunk


class HybridRetrievedChunk(BaseModel):
    """
    Represents a chunk returned by hybrid retrieval.

    The chunk may be discovered by:
    - dense retrieval
    - BM25
    - both
    """

    chunk: RetrievedChunk

    dense_score: float = 0.0
    bm25_score: float = 0.0

    dense_rank: int | None = None
    bm25_rank: int | None = None

    hybrid_score: float = 0.0


class HybridRetriever:
    """
    Combines dense retrieval and BM25 retrieval.

    The current implementation uses Reciprocal Rank Fusion (RRF).

    RRF formula:

        RRF(d) = Σ 1 / (k + rank)

    A document appearing in both retrieval systems receives
    a higher hybrid score.
    """

    DEFAULT_DENSE_TOP_K = 10
    DEFAULT_BM25_TOP_K = 10
    DEFAULT_FINAL_TOP_K = 10

    RRF_K = 60

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> None:

        if dense_top_k < 1:
            raise ValueError("dense_top_k must be at least 1.")

        if bm25_top_k < 1:
            raise ValueError("bm25_top_k must be at least 1.")

        if final_top_k < 1:
            raise ValueError("final_top_k must be at least 1.")

        self.dense_retriever = dense_retriever or DenseRetriever(
            top_k=dense_top_k
        )

        self.bm25_retriever = bm25_retriever

        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

    def retrieve(
        self,
        question: str,
    ) -> list[HybridRetrievedChunk]:
        """
        Retrieve candidates using dense + BM25 retrieval.

        BM25 retriever must be supplied when creating the class.

        Returns:
            List of hybrid-ranked chunks.
        """

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        logger.info(
            "Hybrid retrieval started. "
            f"question={question!r}, "
            f"dense_top_k={self.dense_top_k}, "
            f"bm25_top_k={self.bm25_top_k}, "
            f"final_top_k={self.final_top_k}"
        )

        # ---------------------------------------------------------
        # Dense Retrieval
        # ---------------------------------------------------------

        dense_results = self.dense_retriever.retrieve(
            question,
            top_k=self.dense_top_k,
        )

        # ---------------------------------------------------------
        # BM25 Retrieval
        # ---------------------------------------------------------

        if self.bm25_retriever is None:
            raise RuntimeError(
                "BM25Retriever is required for hybrid retrieval."
            )

        bm25_results = self.bm25_retriever.retrieve(
            question,
            top_k=self.bm25_top_k,
        )

        # ---------------------------------------------------------
        # Fuse Results
        # ---------------------------------------------------------

        fused_results = self._fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        fused_results.sort(
            key=lambda item: item.hybrid_score,
            reverse=True,
        )

        final_results = fused_results[: self.final_top_k]

        logger.info(
            "Hybrid retrieval completed. "
            f"Dense={len(dense_results)}, "
            f"BM25={len(bm25_results)}, "
            f"Final={len(final_results)}"
        )

        return final_results

    def _fuse(
        self,
        dense_results: list[RetrievedChunk],
        bm25_results: list,
    ) -> list[HybridRetrievedChunk]:
        """
        Fuse dense and BM25 rankings using Reciprocal Rank Fusion.

        Important:
        We use chunk_id as the unique identity of a chunk.

        This means if the same chunk appears in both retrieval
        systems, its evidence is combined instead of duplicated.
        """

        candidates: dict[str, HybridRetrievedChunk] = {}

        # ---------------------------------------------------------
        # Dense Results
        # ---------------------------------------------------------

        for rank, chunk in enumerate(dense_results, start=1):

            chunk_id = chunk.chunk_id

            if chunk_id not in candidates:
                candidates[chunk_id] = HybridRetrievedChunk(
                    chunk=chunk,
                )

            result = candidates[chunk_id]

            result.dense_score = float(chunk.score)
            result.dense_rank = rank

            result.hybrid_score += self._rrf_score(rank)

        # ---------------------------------------------------------
        # BM25 Results
        # ---------------------------------------------------------

        for rank, bm25_result in enumerate(bm25_results, start=1):

            chunk = bm25_result.chunk
            chunk_id = chunk.chunk_id

            if chunk_id not in candidates:
                candidates[chunk_id] = HybridRetrievedChunk(
                    chunk=chunk,
                )

            result = candidates[chunk_id]

            result.bm25_score = float(bm25_result.score)
            result.bm25_rank = rank

            result.hybrid_score += self._rrf_score(rank)

        return list(candidates.values())

    def _rrf_score(self, rank: int) -> float:
        """
        Calculate Reciprocal Rank Fusion score.
        """

        return 1.0 / (self.RRF_K + rank)