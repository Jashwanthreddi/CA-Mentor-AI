"""
Manual generation smoke test.

Flow:

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
Gemini Generation
"""

from __future__ import annotations

from app.rag.bm25_corpus import BM25CorpusLoader
from app.rag.bm25_retriever import BM25Retriever
from app.rag.dense_retriever import DenseRetriever
from app.rag.generator import GeminiGenerator
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import BGEReranker


QUESTION = "What is accounting?"


def main() -> None:

    print("\n" + "=" * 70)
    print("CA MENTOR AI - GENERATION SMOKE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD BM25 CORPUS
    # --------------------------------------------------------

    print("\n[1/5] Loading BM25 corpus...")

    bm25_corpus = BM25CorpusLoader().load()

    bm25_retriever = BM25Retriever(
        bm25_corpus
    )

    print(
        f"Loaded {len(bm25_corpus)} chunks."
    )

    # --------------------------------------------------------
    # 2. RETRIEVERS
    # --------------------------------------------------------

    print("\n[2/5] Initializing retrievers...")

    dense_retriever = DenseRetriever(
        top_k=10
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=10,
    )

    # --------------------------------------------------------
    # 3. HYBRID RETRIEVAL
    # --------------------------------------------------------

    print("\n[3/5] Running hybrid retrieval...")

    hybrid_results = hybrid_retriever.retrieve(
        QUESTION
    )

    print(
        f"Hybrid candidates: {len(hybrid_results)}"
    )

    # --------------------------------------------------------
    # 4. RERANK
    # --------------------------------------------------------

    print("\n[4/5] Running reranker...")

    reranker = BGEReranker(
        top_k=5
    )

    reranked_results = reranker.rerank(
        question=QUESTION,
        candidates=hybrid_results,
        top_k=5,
    )

    print(
        f"Reranked evidence: "
        f"{len(reranked_results)}"
    )

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        chunk = result.chunk.chunk

        print(
            f"\n--- Evidence {rank} ---"
        )

        print(
            f"chunk_id: {chunk.chunk_id}"
        )

        print(
            f"score: "
            f"{result.reranker_score:.6f}"
        )

        print(
            f"pages: "
            f"{chunk.page_start}-{chunk.page_end}"
        )

        print(
            f"type: "
            f"{chunk.educational_type}"
        )

        print(
            f"text: "
            f"{chunk.text[:500]}"
        )

    # --------------------------------------------------------
    # 5. GENERATION
    # --------------------------------------------------------

    print("\n[5/5] Generating answer with Gemini...")

    generator = GeminiGenerator()

    result = generator.generate(
        question=QUESTION,
        candidates=reranked_results,
    )

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    print(result.answer)

    print("\n" + "=" * 70)
    print("ATOMIC CLAIMS")
    print("=" * 70)

    for index, claim in enumerate(
        result.atomic_claims,
        start=1,
    ):

        print(
            f"\nClaim {index}:"
        )

        print(
            f"  {claim.claim}"
        )

        print(
            "  Evidence:"
        )

        for chunk_id in (
            claim.evidence_chunk_ids
        ):

            print(
                f"    - {chunk_id}"
            )

    print("\n" + "=" * 70)
    print("EVIDENCE REFERENCES")
    print("=" * 70)

    for evidence in result.evidence:

        print(
            f"\nRank: {evidence.rank}"
        )

        print(
            f"Chunk: {evidence.chunk_id}"
        )

        print(
            f"Reranker score: "
            f"{evidence.reranker_score:.6f}"
        )

        print(
            f"Pages: "
            f"{evidence.page_start}-"
            f"{evidence.page_end}"
        )

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()