"""
Command-line runner for CA Mentor AI generation evaluation.
"""

from __future__ import annotations

import traceback

from app.evaluation.datasets.generation_dataset import (
    get_generation_questions,
)
from app.evaluation.generation.evaluator import (
    GenerationEvaluator,
)
from app.evaluation.generation.metrics import (
    calculate_generation_metrics,
)
from app.rag.bm25_corpus import BM25CorpusLoader
from app.rag.bm25_retriever import BM25Retriever
from app.rag.dense_retriever import DenseRetriever
from app.rag.generator import GeminiGenerator
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import BGEReranker


def main() -> None:

    print()
    print("=" * 70)
    print("CA MENTOR AI - GENERATION EVALUATION")
    print("=" * 70)

    # =============================================================
    # 1. LOAD BM25 CORPUS
    # =============================================================

    print()
    print("[1/5] Loading BM25 corpus...")

    corpus_loader = BM25CorpusLoader()

    corpus = corpus_loader.load()

    print(
        f"Loaded {len(corpus)} chunks."
    )

    if not corpus:
        raise RuntimeError(
            "BM25 corpus is empty."
        )

    bm25_retriever = BM25Retriever(
        chunks=corpus
    )

    # =============================================================
    # 2. INITIALIZE RETRIEVAL
    # =============================================================

    print()
    print("[2/5] Initializing retrieval pipeline...")

    dense_retriever = DenseRetriever()

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    reranker = BGEReranker(
        top_k=5
    )

    # =============================================================
    # 3. INITIALIZE GENERATOR
    # =============================================================

    print()
    print("[3/5] Initializing generator...")

    generator = GeminiGenerator()

    # =============================================================
    # 4. INITIALIZE EVALUATOR
    # =============================================================

    evaluator = GenerationEvaluator(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
    )

    questions = get_generation_questions()

    print()
    print(
        f"[4/5] Evaluating {len(questions)} questions..."
    )

    results = []

    # =============================================================
    # EVALUATE QUESTIONS
    # =============================================================

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print()
        print("-" * 70)
        print(
            f"[{index}/{len(questions)}] "
            f"{question}"
        )
        print("-" * 70)

        try:

            result = evaluator.evaluate(
                question
            )

        except Exception as exc:

            error_message = (
                f"{type(exc).__name__}: {exc}"
            )

            print()
            print("  Generation: EXCEPTION")
            print(
                f"  Error: {error_message}"
            )

            traceback.print_exc()

            results.append(
                {
                    "question": question,
                    "success": False,
                    "claim_count": 0,
                    "claims_without_evidence": 0,
                    "invalid_evidence_references": 0,
                    "latency_seconds": 0.0,
                    "error": error_message,
                }
            )

            continue

        # =========================================================
        # STORE RESULT
        # =========================================================

        results.append(
            {
                "question": result.question,
                "success": result.success,
                "claim_count": result.claim_count,
                "claims_without_evidence": (
                    result.claims_without_evidence
                ),
                "invalid_evidence_references": (
                    result.invalid_evidence_references
                ),
                "latency_seconds": (
                    result.latency_seconds
                ),
                "error": result.error,
            }
        )

        # =========================================================
        # DISPLAY RESULT
        # =========================================================

        if result.success:

            print()
            print("  Generation: PASS")

            print(
                f"  Claims: "
                f"{result.claim_count}"
            )

            print(
                "  Claims without evidence: "
                f"{result.claims_without_evidence}"
            )

            print(
                "  Invalid evidence references: "
                f"{result.invalid_evidence_references}"
            )

            print(
                "  Latency: "
                f"{result.latency_seconds:.2f}s"
            )

        else:

            print()
            print("  Generation: FAIL")

            print(
                f"  Error: {result.error}"
            )

            print(
                "  Latency: "
                f"{result.latency_seconds:.2f}s"
            )

    # =============================================================
    # 5. CALCULATE METRICS
    # =============================================================

    print()
    print("[5/5] Calculating metrics...")

    metrics = calculate_generation_metrics(
        results
    )

    # =============================================================
    # FINAL METRICS
    # =============================================================

    print()
    print("=" * 70)
    print("GENERATION EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Total questions:              "
        f"{metrics.total_questions}"
    )

    print(
        f"Successful generations:       "
        f"{metrics.successful_generations}"
    )

    print(
        f"Failed generations:           "
        f"{metrics.failed_generations}"
    )

    print(
        f"Generation success rate:      "
        f"{metrics.generation_success_rate:.2f}%"
    )

    print(
        f"Total atomic claims:          "
        f"{metrics.total_claims}"
    )

    print(
        f"Claims with evidence:         "
        f"{metrics.claims_with_evidence}"
    )

    print(
        f"Claims without evidence:      "
        f"{metrics.claims_without_evidence}"
    )

    print(
        f"Claim/evidence coverage:      "
        f"{metrics.claim_evidence_coverage:.2f}%"
    )

    print(
        f"Invalid evidence references:  "
        f"{metrics.invalid_evidence_references}"
    )

    print(
        f"Evidence reference validity:  "
        f"{metrics.evidence_reference_validity:.2f}%"
    )

    print(
        f"Average generation latency:   "
        f"{metrics.average_latency_seconds:.2f}s"
    )

    print("=" * 70)

    # =============================================================
    # FINAL DECISION
    # =============================================================

    if (
        metrics.generation_success_rate >= 90.0
        and metrics.claims_without_evidence == 0
        and metrics.invalid_evidence_references == 0
    ):

        print()
        print("RESULT: PASS")

        print(
            "Generation layer passed the "
            "deterministic evaluation."
        )

    else:

        print()
        print("RESULT: NEEDS REVIEW")

        print(
            "Generation layer requires review."
        )

    print()


if __name__ == "__main__":
    main()