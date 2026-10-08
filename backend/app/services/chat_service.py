from __future__ import annotations

from app.core.logging import logger

from app.rag.bm25_corpus import BM25CorpusLoader
from app.rag.bm25_retriever import BM25Retriever
from app.rag.dense_retriever import DenseRetriever
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import BGEReranker
from app.rag.generator import GeminiGenerator

from app.schemas.chat import ChatResponse, ChatSource


class ChatService:
    """
    Main application service for CA Mentor AI.

    RAG pipeline:

        Student Question
              ↓
        Hybrid Retrieval
              ↓
        BGE Reranker
              ↓
        Gemini Generation
              ↓
        Answer + Sources
    """

    def __init__(self) -> None:
        self._bm25_corpus_loader: BM25CorpusLoader | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._dense_retriever: DenseRetriever | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._reranker: BGEReranker | None = None
        self._generator: GeminiGenerator | None = None

    def _initialize(self) -> None:
        """
        Lazily initialize the RAG components.

        Expensive models are loaded only when the first
        chat request arrives.
        """

        if self._hybrid_retriever is not None:
            return

        logger.info("Initializing CA Mentor AI chat service.")

        # ---------------------------------------------------------
        # 1. Load BM25 corpus from Qdrant
        # ---------------------------------------------------------

        logger.info("Loading BM25 corpus.")

        self._bm25_corpus_loader = BM25CorpusLoader()

        corpus = self._bm25_corpus_loader.load()

        self._bm25_retriever = BM25Retriever(
            chunks=corpus
        )

        # ---------------------------------------------------------
        # 2. Dense retriever
        # ---------------------------------------------------------

        self._dense_retriever = DenseRetriever()

        # ---------------------------------------------------------
        # 3. Hybrid retriever
        # ---------------------------------------------------------

        self._hybrid_retriever = HybridRetriever(
            dense_retriever=self._dense_retriever,
            bm25_retriever=self._bm25_retriever,
        )

        # ---------------------------------------------------------
        # 4. BGE reranker
        # ---------------------------------------------------------

        self._reranker = BGEReranker()

        # ---------------------------------------------------------
        # 5. Gemini generator
        # ---------------------------------------------------------

        self._generator = GeminiGenerator()

        logger.info(
            "CA Mentor AI chat service initialized successfully."
        )

    def chat(self, question: str) -> ChatResponse:
        """
        Execute the complete CA Mentor AI RAG pipeline.
        """

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        logger.info(
            "CA Mentor AI chat started. question=%r",
            question,
        )

        # Initialize components if this is the first request.
        self._initialize()

        # These assertions make the types explicit after lazy
        # initialization and protect against accidental misuse.
        assert self._hybrid_retriever is not None
        assert self._reranker is not None
        assert self._generator is not None

        # =========================================================
        # 1. HYBRID RETRIEVAL
        # =========================================================

        hybrid_results = self._hybrid_retriever.retrieve(
            question=question,
        )

        logger.info(
            "Hybrid retrieval returned %d candidates.",
            len(hybrid_results),
        )

        if not hybrid_results:
            return ChatResponse(
                question=question,
                answer=(
                    "I could not find relevant information "
                    "in the CA study material."
                ),
                sources=[],
            )

        # =========================================================
        # 2. BGE RERANKING
        # =========================================================

        reranked_results = self._reranker.rerank(
            question=question,
            candidates=hybrid_results,
            top_k=5,
        )

        logger.info(
            "Reranker returned %d evidence chunks.",
            len(reranked_results),
        )

        if not reranked_results:
            return ChatResponse(
                question=question,
                answer=(
                    "I could not find sufficiently relevant "
                    "information in the CA study material."
                ),
                sources=[],
            )

        # =========================================================
        # 3. GEMINI GENERATION
        # =========================================================

        generation_result = self._generator.generate(
            question=question,
            candidates=reranked_results,
        )

        # =========================================================
        # 4. BUILD SOURCE INFORMATION
        # =========================================================

        sources = [
            ChatSource(
                chunk_id=evidence.chunk_id,
                rank=evidence.rank,
                reranker_score=evidence.reranker_score,
                page_start=evidence.page_start,
                page_end=evidence.page_end,
            )
            for evidence in generation_result.evidence
        ]

        # =========================================================
        # 5. RETURN FINAL RESPONSE
        # =========================================================

        logger.info(
            "CA Mentor AI chat completed successfully."
        )

        return ChatResponse(
            question=question,
            answer=generation_result.answer,
            sources=sources,
        )