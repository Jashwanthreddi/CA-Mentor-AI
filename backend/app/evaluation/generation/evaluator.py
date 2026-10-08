"""
Generation evaluation for CA Mentor AI.

This module evaluates the generation layer independently from the
production RAG API.

Evaluation flow:

Question
    ↓
Hybrid Retrieval
    ↓
BGE Reranking
    ↓
Gemini Generation
    ↓
Deterministic Structural Validation
    ↓
GenerationEvaluationResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.rag.generator import GeminiGenerator
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import BGEReranker


@dataclass
class GenerationEvaluationResult:
    """
    Result of evaluating one generation request.
    """

    question: str
    success: bool

    claim_count: int

    claims_without_evidence: int

    invalid_evidence_references: int

    latency_seconds: float

    error: str | None = None


class GenerationEvaluator:
    """
    Evaluates the generation pipeline using deterministic checks.

    Current checks:

    1. Generation succeeds.
    2. Atomic claims are produced.
    3. Claims reference evidence.
    4. Evidence references point to retrieved evidence.
    5. Generation latency is recorded.

    Semantic entailment is intentionally not evaluated here.
    That can be added later using an independent NLI model or
    LLM-based judge.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: BGEReranker,
        generator: GeminiGenerator,
    ) -> None:

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator

    def evaluate(
        self,
        question: str,
    ) -> GenerationEvaluationResult:

        start_time = time.perf_counter()

        try:

            # =====================================================
            # 1. HYBRID RETRIEVAL
            # =====================================================

            hybrid_results = self.hybrid_retriever.retrieve(
                question=question
            )

            if not hybrid_results:
                raise RuntimeError(
                    "Hybrid retrieval returned no candidates."
                )

            # =====================================================
            # 2. RERANKING
            # =====================================================

            reranked_results = self.reranker.rerank(
                question=question,
                candidates=hybrid_results,
                top_k=5,
            )

            if not reranked_results:
                raise RuntimeError(
                    "Reranker returned no candidates."
                )

            # =====================================================
            # 3. GENERATION
            # =====================================================

            generation_result = self.generator.generate(
                question=question,
                candidates=reranked_results,
            )

            if generation_result is None:
                raise RuntimeError(
                    "Generator returned None."
                )

            # =====================================================
            # 4. VALIDATE ATOMIC CLAIMS
            # =====================================================

            claim_count = len(
                generation_result.atomic_claims
            )

            if claim_count == 0:
                raise RuntimeError(
                    "Generator returned zero atomic claims."
                )

            # =====================================================
            # 5. EXTRACT VALID EVIDENCE IDS
            # =====================================================
            #
            # Actual object structure:
            #
            # RerankedChunk
            #     ↓
            # .chunk
            #     ↓
            # HybridRetrievedChunk
            #     ↓
            # .chunk
            #     ↓
            # RetrievedChunk
            #     ↓
            # .chunk_id
            #
            # Therefore:
            #
            # item.chunk.chunk.chunk_id
            #
            # =====================================================

            valid_evidence_ids = {
                item.chunk.chunk.chunk_id
                for item in reranked_results
            }

            # =====================================================
            # 6. VALIDATE CLAIM → EVIDENCE LINKS
            # =====================================================

            claims_without_evidence = 0
            invalid_evidence_references = 0

            for claim in generation_result.atomic_claims:

                evidence_ids = claim.evidence_chunk_ids

                # -------------------------------------------------
                # Claim has no evidence
                # -------------------------------------------------

                if not evidence_ids:
                    claims_without_evidence += 1
                    continue

                # -------------------------------------------------
                # Validate every evidence reference
                # -------------------------------------------------

                for evidence_id in evidence_ids:

                    if evidence_id not in valid_evidence_ids:
                        invalid_evidence_references += 1

            # =====================================================
            # 7. LATENCY
            # =====================================================

            latency_seconds = (
                time.perf_counter() - start_time
            )

            # =====================================================
            # 8. DETERMINE SUCCESS
            # =====================================================

            success = (
                claims_without_evidence == 0
                and invalid_evidence_references == 0
            )

            return GenerationEvaluationResult(
                question=question,
                success=success,
                claim_count=claim_count,
                claims_without_evidence=(
                    claims_without_evidence
                ),
                invalid_evidence_references=(
                    invalid_evidence_references
                ),
                latency_seconds=latency_seconds,
                error=None,
            )

        except Exception as exc:

            latency_seconds = (
                time.perf_counter() - start_time
            )

            error_message = (
                f"{type(exc).__name__}: {exc}"
            )

            return GenerationEvaluationResult(
                question=question,
                success=False,
                claim_count=0,
                claims_without_evidence=0,
                invalid_evidence_references=0,
                latency_seconds=latency_seconds,
                error=error_message,
            )