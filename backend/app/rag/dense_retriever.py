"""
Dense Retriever for CA Mentor AI.

Pipeline
--------
User Question
    ↓
BGE Embedding
    ↓
Qdrant Dense Search
    ↓
Retrieved Chunks
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.embedder import DocumentEmbedder


class RetrievedChunk(BaseModel):
    """One chunk retrieved from Qdrant."""

    chunk_id: str
    text: str
    score: float

    # Document provenance
    document_id: str | None = None
    source: str | None = None
    edition: str | None = None

    # Academic hierarchy
    level: str | None = None
    paper: str | None = None
    module: str | None = None
    chapter: str | None = None
    unit: str | None = None
    topic: str | None = None
    subtopic: str | None = None

    # Educational metadata
    educational_type: str | None = None
    difficulty: str | None = None
    teaching_priority: str | None = None
    embedding_strategy: str | None = None

    # Provenance
    page_start: int | None = None
    page_end: int | None = None

    # Content flags
    has_table: bool = False
    has_image: bool = False
    has_formula: bool = False
    has_example: bool = False

    # Additional fields
    eku_id: str | None = None
    chunk_index: int | None = None
    total_chunks: int | None = None
    token_count: int | None = None
    metadata_version: str | None = None


class DenseRetriever:
    """
    Retrieve relevant chunks from Qdrant using dense embeddings.
    """

    DEFAULT_TOP_K = 5

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: DocumentEmbedder | None = None,
        collection_name: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.collection_name = (
            collection_name
            if collection_name is not None
            else settings.QDRANT_COLLECTION
        )

        if not self.collection_name.strip():
            raise ValueError(
                "QDRANT_COLLECTION is not configured."
            )

        self.top_k = top_k
        self._client = client
        self._embedder = embedder

    @property
    def client(self) -> QdrantClient:
        """Lazily create Qdrant client."""

        if self._client is None:

            if not settings.QDRANT_URL:
                raise ValueError(
                    "QDRANT_URL is not configured."
                )

            if not settings.QDRANT_API_KEY:
                raise ValueError(
                    "QDRANT_API_KEY is not configured."
                )

            logger.info(
                "Initializing Qdrant client for retrieval."
            )

            self._client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                prefer_grpc=False,
                timeout=60,
            )

            logger.info(
                "Qdrant retrieval client initialized."
            )

        return self._client

    @property
    def embedder(self) -> DocumentEmbedder:
        """Lazily create the query embedding service."""

        if self._embedder is None:

            logger.info(
                "Initializing query embedding service."
            )

            self._embedder = DocumentEmbedder(
                model_name=settings.EMBEDDING_MODEL,
                batch_size=1,
                device="cpu",
            )

        return self._embedder

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a question.
        """

        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        limit = (
            top_k
            if top_k is not None
            else self.top_k
        )

        if limit < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        logger.info(
            f"Dense retrieval started. "
            f"question={question!r}, top_k={limit}"
        )

        query_vector = self._embed_query(question)

        results = self._search_qdrant(
            query_vector=query_vector,
            limit=limit,
        )

        retrieved_chunks = [
            self._convert_result(result)
            for result in results
        ]

        logger.info(
            f"Dense retrieval completed. "
            f"Retrieved {len(retrieved_chunks)} chunks."
        )

        return retrieved_chunks

    def _embed_query(
        self,
        question: str,
    ) -> list[float]:
        """
        Generate the embedding for the user question.

        Uses the same BGE model used during ingestion.
        """

        logger.info(
            "Generating embedding for user question."
        )

        model = self.embedder.model

        embeddings = model.encode(
            [question],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        if len(embeddings) != 1:
            raise RuntimeError(
                "Query embedding model returned "
                "an unexpected number of embeddings."
            )

        vector = [
            float(value)
            for value in embeddings[0]
        ]

        if len(vector) != DocumentEmbedder.EXPECTED_DIMENSION:
            raise ValueError(
                "Unexpected query embedding dimension: "
                f"expected "
                f"{DocumentEmbedder.EXPECTED_DIMENSION}, "
                f"got {len(vector)}."
            )

        return vector

    def _search_qdrant(
        self,
        query_vector: list[float],
        limit: int,
    ) -> list[Any]:
        """Search Qdrant using cosine similarity."""

        logger.info(
            f"Searching Qdrant collection "
            f"'{self.collection_name}' "
            f"with limit={limit}."
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return response.points

    def _convert_result(
        self,
        result: Any,
    ) -> RetrievedChunk:
        """Convert a Qdrant result into RetrievedChunk."""

        payload = result.payload or {}

        return RetrievedChunk(
            chunk_id=str(
                payload.get(
                    "chunk_id",
                    result.id,
                )
            ),
            text=str(
                payload.get(
                    "text",
                    "",
                )
            ),
            score=float(result.score),

            document_id=payload.get("document_id"),
            source=payload.get("source"),
            edition=payload.get("edition"),

            level=payload.get("level"),
            paper=payload.get("paper"),
            module=payload.get("module"),
            chapter=payload.get("chapter"),
            unit=payload.get("unit"),
            topic=payload.get("topic"),
            subtopic=payload.get("subtopic"),

            educational_type=payload.get(
                "educational_type"
            ),
            difficulty=payload.get(
                "difficulty"
            ),
            teaching_priority=payload.get(
                "teaching_priority"
            ),
            embedding_strategy=payload.get(
                "embedding_strategy"
            ),

            page_start=payload.get(
                "page_start"
            ),
            page_end=payload.get(
                "page_end"
            ),

            has_table=bool(
                payload.get("has_table", False)
            ),
            has_image=bool(
                payload.get("has_image", False)
            ),
            has_formula=bool(
                payload.get("has_formula", False)
            ),
            has_example=bool(
                payload.get("has_example", False)
            ),

            eku_id=payload.get("eku_id"),
            chunk_index=payload.get("chunk_index"),
            total_chunks=payload.get("total_chunks"),
            token_count=payload.get("token_count"),
            metadata_version=payload.get("version"),
        )