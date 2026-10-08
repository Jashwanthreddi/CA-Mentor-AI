"""
Retrieval evaluation for CA Mentor AI.

Metrics:
    - Hit Rate@K
    - Recall@K
    - Precision@K
    - MRR@K

This module evaluates retrieved chunk IDs against a manually
curated evaluation dataset.

It does not modify the retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.evaluation_dataset import RetrievalEvaluationCase


@dataclass(frozen=True)
class RetrievalMetrics:
    """
    Metrics for one retrieval evaluation case.
    """

    question: str
    hit: bool
    recall: float
    precision: float
    reciprocal_rank: float
    first_relevant_rank: int | None


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """
    Complete evaluation report.
    """

    total_questions: int

    hit_rate_at_k: float
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float

    results: tuple[RetrievalMetrics, ...]


class RetrievalEvaluator:
    """
    Evaluates retrieval results using standard information
    retrieval metrics.
    """

    def evaluate(
        self,
        cases: tuple[RetrievalEvaluationCase, ...],
        retrieved_chunk_ids: dict[str, list[str]],
        k: int,
    ) -> RetrievalEvaluationReport:

        if k < 1:
            raise ValueError("k must be at least 1.")

        if not cases:
            raise ValueError("cases cannot be empty.")

        results: list[RetrievalMetrics] = []

        for case in cases:
            retrieved = retrieved_chunk_ids.get(
                case.question,
                [],
            )

            metrics = self._evaluate_case(
                case=case,
                retrieved_chunk_ids=retrieved,
                k=k,
            )

            results.append(metrics)

        total = len(results)

        hit_rate = (
            sum(result.hit for result in results)
            / total
        )

        recall = (
            sum(result.recall for result in results)
            / total
        )

        precision = (
            sum(result.precision for result in results)
            / total
        )

        mrr = (
            sum(result.reciprocal_rank for result in results)
            / total
        )

        return RetrievalEvaluationReport(
            total_questions=total,
            hit_rate_at_k=hit_rate,
            recall_at_k=recall,
            precision_at_k=precision,
            mrr_at_k=mrr,
            results=tuple(results),
        )

    def _evaluate_case(
        self,
        case: RetrievalEvaluationCase,
        retrieved_chunk_ids: list[str],
        k: int,
    ) -> RetrievalMetrics:

        top_k = retrieved_chunk_ids[:k]

        relevant = case.relevant_chunk_ids

        relevant_retrieved = [
            chunk_id
            for chunk_id in top_k
            if chunk_id in relevant
        ]

        hit = len(relevant_retrieved) > 0

        recall = (
            len(set(relevant_retrieved))
            / len(relevant)
        )

        precision = (
            len(relevant_retrieved)
            / len(top_k)
            if top_k
            else 0.0
        )

        first_relevant_rank: int | None = None

        for rank, chunk_id in enumerate(
            top_k,
            start=1,
        ):
            if chunk_id in relevant:
                first_relevant_rank = rank
                break

        reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        return RetrievalMetrics(
            question=case.question,
            hit=hit,
            recall=recall,
            precision=precision,
            reciprocal_rank=reciprocal_rank,
            first_relevant_rank=first_relevant_rank,
        )