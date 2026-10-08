"""
BM25 corpus loader for CA Mentor AI.

Loads the complete chunk corpus from Qdrant and converts each
stored payload into a RetrievedChunk.

This module is intentionally separate from BM25Retriever so that
BM25 retrieval has an independent corpus from dense retrieval.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.logging import logger
from app.rag.dense_retriever import RetrievedChunk


class BM25CorpusLoader:
    """Load the complete retrieval corpus from Qdrant."""

    DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self._client = client
        self.collection_name = (
            collection_name or settings.QDRANT_COLLECTION
        )
        self.batch_size = batch_size

    @property
    def client(self) -> QdrantClient:
        """Create the Qdrant client lazily."""

        if self._client is None:
            if not settings.QDRANT_URL:
                raise ValueError("QDRANT_URL is not configured.")

            if not settings.QDRANT_API_KEY:
                raise ValueError("QDRANT_API_KEY is not configured.")

            logger.info(
                "Initializing Qdrant client for BM25 corpus loading."
            )

            self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False,
            timeout=60,
        )

        return self._client

    def load(self) -> list[RetrievedChunk]:
        """
        Load every chunk stored in the Qdrant collection.

        Pagination is handled using Qdrant's scroll API.
        """

        logger.info(
            f"Loading complete BM25 corpus from collection "
            f"'{self.collection_name}'."
        )

        chunks: list[RetrievedChunk] = []
        offset: Any = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=self.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                chunk = self._payload_to_chunk(payload)

                if chunk is not None:
                    chunks.append(chunk)

            if next_offset is None:
                break

            offset = next_offset

        if not chunks:
            raise RuntimeError(
                f"No chunks found in Qdrant collection "
                f"'{self.collection_name}'."
            )

        logger.info(
            f"BM25 corpus loaded successfully. "
            f"Total chunks={len(chunks)}."
        )

        return chunks

    @staticmethod
    def _payload_to_chunk(
        payload: dict[str, Any],
    ) -> RetrievedChunk | None:
        """Convert a Qdrant payload into a RetrievedChunk."""

        chunk_id = payload.get("chunk_id")
        text = payload.get("text")

        if not chunk_id:
            logger.warning(
                "Skipping Qdrant point without chunk_id."
            )
            return None

        if not text or not str(text).strip():
            logger.warning(
                f"Skipping chunk '{chunk_id}' because text is empty."
            )
            return None

        return RetrievedChunk(
            chunk_id=str(chunk_id),
            text=str(text),
            score=0.0,
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
            educational_type=payload.get("educational_type"),
            difficulty=payload.get("difficulty"),
            teaching_priority=payload.get("teaching_priority"),
            embedding_strategy=payload.get("embedding_strategy"),
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
            has_table=bool(payload.get("has_table", False)),
            has_image=bool(payload.get("has_image", False)),
            has_formula=bool(payload.get("has_formula", False)),
            has_example=bool(payload.get("has_example", False)),
            eku_id=payload.get("eku_id"),
            chunk_index=payload.get("chunk_index"),
            total_chunks=payload.get("total_chunks"),
            token_count=payload.get("token_count"),
            metadata_version=payload.get("version"),
        )