"""
End-to-end retrieval evaluation for CA Mentor AI.

Pipeline evaluated:

Question
    ↓
Dense Retrieval
    ↓
BM25 Retrieval
    ↓
Hybrid Retrieval
    ↓
BGE Reranker
    ↓
Retrieval Metrics

Metrics:
    - Hit Rate@K
    - Recall@K
    - Precision@K
    - MRR@K

This module is evaluation-only.
It does not modify the retrieval pipeline.
"""

from __future__ import annotations
from app.rag.bm25_corpus import BM25CorpusLoader
from app.core.logging import logger
from app.rag.bm25_retriever import BM25Retriever
from app.rag.dense_retriever import DenseRetriever
from app.rag.evaluation_dataset import EVALUATION_DATASET
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import BGEReranker
from app.rag.retrieval_evaluator import RetrievalEvaluator


def main() -> None:
    print("\n" + "=" * 80)
    print("CA MENTOR AI - RETRIEVAL EVALUATION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Dense retrieval
    # ------------------------------------------------------------------

    print("\n[1/4] Initializing dense retriever...")

    dense_retriever = DenseRetriever(top_k=10)

    print("Dense retriever initialized.")

    # ------------------------------------------------------------------
    # 2. Build BM25 corpus
    # ------------------------------------------------------------------
    #
    # The current BM25Retriever expects RetrievedChunk objects.
    # Therefore we first retrieve the available chunks from the dense
    # retriever and use them to construct the BM25 index.
    #
    # This is the current evaluation setup and does not modify the
    # existing BM25 implementation.
    # ------------------------------------------------------------------

    print("\n[2/4] Building BM25 index...")

    # Use a broad retrieval query to obtain the current collection
    # contents through the existing retriever.
    # [2/4] Build independent BM25 corpus from Qdrant
    bm25_corpus = BM25CorpusLoader().load()

    bm25_retriever = BM25Retriever(bm25_corpus)

    print(f"BM25 corpus size: {len(bm25_corpus)}")

    print(f"BM25 index initialized with {len(bm25_corpus)} chunks.")
    # ------------------------------------------------------------------
    # 3. Hybrid retriever
    # ------------------------------------------------------------------

    print("\n[3/4] Initializing hybrid retriever...")

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=10,
    )

    print("Hybrid retriever initialized.")

    # ------------------------------------------------------------------
    # 4. Reranker
    # ------------------------------------------------------------------

    print("\n[4/4] Initializing BGE reranker...")

    reranker = BGEReranker(
        top_k=5,
    )

    print("BGE reranker initialized.")

    # ------------------------------------------------------------------
    # Evaluation storage
    # ------------------------------------------------------------------

    dense_results_by_question: dict[str, list[str]] = {}
    bm25_results_by_question: dict[str, list[str]] = {}
    hybrid_results_by_question: dict[str, list[str]] = {}
    reranked_results_by_question: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Run retrieval for every evaluation question
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RUNNING RETRIEVAL")
    print("=" * 80)

    for index, case in enumerate(EVALUATION_DATASET, start=1):

        question = case.question

        print(f"\n[{index}/{len(EVALUATION_DATASET)}] {question}")
        print("-" * 80)

        # --------------------------------------------------------------
        # Dense
        # --------------------------------------------------------------

        dense_results = dense_retriever.retrieve(
            question,
            top_k=10,
        )

        dense_ids = [
            result.chunk_id
            for result in dense_results
        ]

        dense_results_by_question[question] = dense_ids

        print("\nDense Top 10:")
        for rank, result in enumerate(dense_results, start=1):
            print(
                f"  {rank:>2}. "
                f"score={result.score:.4f} | "
                f"{result.chunk_id} | "
                f"pages={result.page_start}-{result.page_end}"
            )

        # --------------------------------------------------------------
        # BM25
        # --------------------------------------------------------------

        bm25_results = bm25_retriever.retrieve(
            question,
            top_k=10,
        )

        bm25_ids = [
            result.chunk.chunk_id
            for result in bm25_results
        ]

        bm25_results_by_question[question] = bm25_ids

        print("\nBM25 Top 10:")
        for rank, result in enumerate(bm25_results, start=1):
            print(
                f"  {rank:>2}. "
                f"score={result.score:.4f} | "
                f"{result.chunk.chunk_id} | "
                f"pages={result.chunk.page_start}-"
                f"{result.chunk.page_end}"
            )

        # --------------------------------------------------------------
        # Hybrid
        # --------------------------------------------------------------

        hybrid_results = hybrid_retriever.retrieve(
            question,
        )

        hybrid_ids = [
            result.chunk.chunk_id
            for result in hybrid_results
        ]

        hybrid_results_by_question[question] = hybrid_ids

        print("\nHybrid Top 10:")
        for rank, result in enumerate(hybrid_results, start=1):
            print(
                f"  {rank:>2}. "
                f"rrf={result.hybrid_score:.6f} | "
                f"{result.chunk.chunk_id} | "
                f"pages={result.chunk.page_start}-"
                f"{result.chunk.page_end}"
            )

        # --------------------------------------------------------------
        # Reranker
        # --------------------------------------------------------------

        reranked_results = reranker.rerank(
            question,
            hybrid_results,
            top_k=5,
        )

        reranked_ids = [
            result.chunk.chunk.chunk_id
            for result in reranked_results
        ]

        reranked_results_by_question[question] = reranked_ids

        print("\nReranked Top 5:")
        for rank, result in enumerate(reranked_results, start=1):
            print(
                f"  {rank:>2}. "
                f"score={result.reranker_score:.4f} | "
                f"{result.chunk.chunk.chunk_id} | "
                f"pages={result.chunk.chunk.page_start}-"
                f"{result.chunk.chunk.page_end}"
            )

    # ------------------------------------------------------------------
    # Evaluate each stage
    # ------------------------------------------------------------------

    evaluator = RetrievalEvaluator()

    print("\n" + "=" * 80)
    print("RETRIEVAL METRICS")
    print("=" * 80)

    k = 5

    stages = {
        "Dense": dense_results_by_question,
        "BM25": bm25_results_by_question,
        "Hybrid": hybrid_results_by_question,
        "Reranker": reranked_results_by_question,
    }

    reports = {}

    for stage_name, results in stages.items():

        report = evaluator.evaluate(
            cases=EVALUATION_DATASET,
            retrieved_chunk_ids=results,
            k=k,
        )

        reports[stage_name] = report

        print(f"\n{stage_name} @ {k}")
        print("-" * 40)

        print(
            f"Hit Rate@{k}:  "
            f"{report.hit_rate_at_k:.4f}"
        )

        print(
            f"Recall@{k}:     "
            f"{report.recall_at_k:.4f}"
        )

        print(
            f"Precision@{k}:  "
            f"{report.precision_at_k:.4f}"
        )

        print(
            f"MRR@{k}:        "
            f"{report.mrr_at_k:.4f}"
        )

    # ------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 80)

    print(
        f"\n{'Stage':<12}"
        f"{'Hit@5':>10}"
        f"{'Recall@5':>12}"
        f"{'Precision@5':>14}"
        f"{'MRR@5':>10}"
    )

    print("-" * 60)

    for stage_name, report in reports.items():

        print(
            f"{stage_name:<12}"
            f"{report.hit_rate_at_k:>10.4f}"
            f"{report.recall_at_k:>12.4f}"
            f"{report.precision_at_k:>14.4f}"
            f"{report.mrr_at_k:>10.4f}"
        )

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()