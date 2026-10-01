"""
Qdrant uploader for CA Mentor AI.

Responsibilities:

1. Connect to Qdrant.
2. Ensure the target collection exists.
3. Convert EmbeddedChunk objects into Qdrant points.
4. Preserve chunk text and metadata in the payload.
5. Use deterministic point identity based on chunk_id.
6. Upsert points idempotently.
7. Return an IngestionReport.

The uploader does NOT:
- parse PDFs
- classify content
- create chunks
- generate embeddings
- perform retrieval
- perform reranking
- call an LLM
"""

from __future__ import annotations

from time import perf_counter
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.models import (
    EmbeddedChunk,
    IngestionReport,
)


class QdrantUploader:
    """
    Upload embedded educational chunks to Qdrant.

    Application chunk ID:

        chunk-<UUID>

    Qdrant point ID:

        <UUID>

    Example:

        Application:
            chunk-835a5565-c46a-4c85-ba67-41e3fbcd08e5

        Qdrant:
            835a5565-c46a-4c85-ba67-41e3fbcd08e5

    The original application chunk_id is preserved
    inside the Qdrant payload.
    """

    DEFAULT_COLLECTION_NAME = "ca_mentor_chunks"
    VECTOR_DIMENSION = 768

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        vector_dimension: int = VECTOR_DIMENSION,
    ) -> None:
        """
        Initialize the Qdrant uploader.
        """

        if vector_dimension < 1:
            raise ValueError(
                "vector_dimension must be at least 1."
            )

        if collection_name is not None:
            if not collection_name.strip():
                raise ValueError(
                    "collection_name cannot be empty."
                )

            self.collection_name = collection_name

        else:
            self.collection_name = settings.QDRANT_COLLECTION

            if not self.collection_name.strip():
                raise ValueError(
                    "collection_name cannot be empty."
                )

        self.vector_dimension = vector_dimension
        self._client = client

    @property
    def client(self) -> QdrantClient:
        """
        Lazily create and return the Qdrant client.
        """

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
                "Initializing Qdrant client."
            )

            self._client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )

            logger.info(
                "Qdrant client initialized successfully."
            )

        return self._client

    def ensure_collection(self) -> None:
        """
        Ensure the Qdrant collection exists.

        Existing collection:
            Do nothing.

        Missing collection:
            Create it using:
                vector size = vector_dimension
                distance = COSINE
        """

        exists = self.client.collection_exists(
            collection_name=self.collection_name
        )

        if exists:
            logger.info(
                f"Qdrant collection already exists: "
                f"{self.collection_name}"
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dimension,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            f"Created Qdrant collection: "
            f"{self.collection_name}"
        )

    def _extract_uuid(
        self,
        chunk_id: str,
    ) -> UUID:
        """
        Convert an application chunk ID into a UUID.

        Expected format:

            chunk-<UUID>

        Example:

            chunk-835a5565-c46a-4c85-ba67-41e3fbcd08e5
        """

        prefix = "chunk-"

        if not chunk_id.startswith(prefix):
            raise ValueError(
                "Invalid chunk ID format. "
                "Expected 'chunk-<UUID>'."
            )

        uuid_value = chunk_id[len(prefix):]

        try:
            return UUID(uuid_value)

        except ValueError as exc:
            raise ValueError(
                f"Invalid chunk ID format: {chunk_id}. "
                "Chunk ID must contain a valid UUID."
            ) from exc

    def _build_payload(
        self,
        chunk: EmbeddedChunk,
    ) -> dict:
        """
        Build the Qdrant payload.
        """

        metadata = chunk.metadata

        return {
            "chunk_id": chunk.chunk_id,
            "eku_id": chunk.eku_id,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "text": chunk.text,
            "token_count": chunk.token_count,

            # Academic hierarchy
            "level": metadata.level.value,
            "paper": metadata.paper,
            "module": metadata.module,
            "chapter": metadata.chapter,
            "unit": metadata.unit,
            "topic": metadata.topic,
            "subtopic": metadata.subtopic,

            # Educational metadata
            "educational_type": (
                metadata.educational_type.value
            ),
            "difficulty": (
                metadata.difficulty.value
                if metadata.difficulty is not None
                else None
            ),
            "teaching_priority": (
                metadata.teaching_priority.value
                if metadata.teaching_priority is not None
                else None
            ),
            "embedding_strategy": (
                metadata.embedding_strategy.value
            ),

            # Provenance
            "page_start": metadata.page_start,
            "page_end": metadata.page_end,

            # Content flags
            "has_table": metadata.has_table,
            "has_image": metadata.has_image,
            "has_formula": metadata.has_formula,
            "has_example": metadata.has_example,

            # Metadata version
            "version": metadata.version,
        }

    def _build_point(
        self,
        chunk: EmbeddedChunk,
    ) -> PointStruct:
        """
        Convert an EmbeddedChunk into a Qdrant PointStruct.
        """

        point_uuid = self._extract_uuid(
            chunk.chunk_id
        )

        if len(chunk.embedding) != self.vector_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for "
                f"{chunk.chunk_id}: expected "
                f"{self.vector_dimension}, got "
                f"{len(chunk.embedding)}."
            )

        if not all(
            isinstance(value, (int, float))
            for value in chunk.embedding
        ):
            raise TypeError(
                f"Embedding for {chunk.chunk_id} "
                "must contain only numeric values."
            )

        payload = self._build_payload(chunk)

        return PointStruct(
            id=str(point_uuid),
            vector=chunk.embedding,
            payload=payload,
        )

    def upload(
        self,
        chunks: list[EmbeddedChunk],
        *,
        filename: str,
        total_pages: int,
        sections: int,
        eku_count: int,
        tables: int,
        images: int,
        embedding_time: float,
    ) -> IngestionReport:
        """
        Upload embedded chunks to Qdrant.

        Qdrant upsert is used intentionally.

        If the same chunk ID is uploaded again,
        the existing point is replaced instead of
        creating a duplicate.
        """

        if not chunks:
            logger.info(
                "Uploader received no chunks."
            )

            return IngestionReport(
                filename=filename,
                total_pages=total_pages,
                sections=sections,
                eku_count=eku_count,
                chunks=0,
                tables=tables,
                images=images,
                embedding_time=embedding_time,
                upload_time=0.0,
            )

        self.ensure_collection()

        points = [
            self._build_point(chunk)
            for chunk in chunks
        ]

        upload_start = perf_counter()

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        upload_time = (
            perf_counter() - upload_start
        )

        logger.info(
            f"Uploaded {len(points)} chunks to "
            f"Qdrant collection "
            f"'{self.collection_name}' "
            f"in {upload_time:.3f}s."
        )

        return IngestionReport(
            filename=filename,
            total_pages=total_pages,
            sections=sections,
            eku_count=eku_count,
            chunks=len(points),
            tables=tables,
            images=images,
            embedding_time=embedding_time,
            upload_time=upload_time,
        )