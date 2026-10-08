"""
Deterministic metrics for CA Mentor AI generation evaluation.

These metrics intentionally avoid an LLM judge.

They measure:
    - generation success
    - claim production
    - claim/evidence coverage
    - evidence reference validity
    - generation latency

Semantic faithfulness and correctness are intentionally
left for the later self-grading/NLI evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationMetrics:
    """Aggregated generation evaluation metrics."""

    total_questions: int
    successful_generations: int
    failed_generations: int

    total_claims: int
    claims_with_evidence: int
    claims_without_evidence: int

    invalid_evidence_references: int

    generation_success_rate: float
    claim_evidence_coverage: float
    evidence_reference_validity: float

    average_latency_seconds: float


def calculate_generation_metrics(
    results: list[dict],
) -> GenerationMetrics:
    """
    Calculate deterministic generation metrics.

    Expected result fields:

        success
        claim_count
        claims_without_evidence
        invalid_evidence_references
        latency_seconds
    """

    total_questions = len(results)

    successful_generations = sum(
        1
        for result in results
        if result["success"]
    )

    failed_generations = (
        total_questions
        - successful_generations
    )

    total_claims = sum(
        result["claim_count"]
        for result in results
    )

    claims_without_evidence = sum(
        result["claims_without_evidence"]
        for result in results
    )

    claims_with_evidence = (
        total_claims
        - claims_without_evidence
    )

    invalid_evidence_references = sum(
        result["invalid_evidence_references"]
        for result in results
    )

    generation_success_rate = (
        successful_generations
        / total_questions
        * 100
        if total_questions
        else 0.0
    )

    claim_evidence_coverage = (
        claims_with_evidence
        / total_claims
        * 100
        if total_claims
        else 0.0
    )

    evidence_reference_validity = (
        100.0
        if invalid_evidence_references == 0
        else (
            0.0
            if total_claims == 0
            else 100.0
            * (
                1
                - invalid_evidence_references
                / total_claims
            )
        )
    )

    successful_latencies = [
        result["latency_seconds"]
        for result in results
        if result["success"]
    ]

    average_latency_seconds = (
        sum(successful_latencies)
        / len(successful_latencies)
        if successful_latencies
        else 0.0
    )

    return GenerationMetrics(
        total_questions=total_questions,
        successful_generations=successful_generations,
        failed_generations=failed_generations,
        total_claims=total_claims,
        claims_with_evidence=claims_with_evidence,
        claims_without_evidence=claims_without_evidence,
        invalid_evidence_references=invalid_evidence_references,
        generation_success_rate=generation_success_rate,
        claim_evidence_coverage=claim_evidence_coverage,
        evidence_reference_validity=evidence_reference_validity,
        average_latency_seconds=average_latency_seconds,
    )