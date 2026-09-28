"""
Embedding Service

Purpose
-------
Convert validated DocumentChunk objects into EmbeddedChunk objects
using a local Sentence Transformer model.

Default model
-------------
BAAI/bge-base-en-v1.5

The embedding model is kept behind this module so the rest of the
ingestion pipeline does not depend directly on Sentence Transformers.

Input
-----
list[DocumentChunk]

Output
------
list[EmbeddedChunk]

Responsibilities
----------------
- Load the embedding model.
- Embed chunk text in batches.
- Preserve chunk information and metadata.
- Validate embedding dimensions.
- Avoid external API calls.

The embedder does NOT:
- modify chunk text
- perform retrieval
- access Qdrant
- perform BM25 search
- perform reranking
"""

from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.ingestion.models import DocumentChunk, EmbeddedChunk


class DocumentEmbedder:
    """
    Generate vector embeddings for document chunks.

    Parameters
    ----------
    model_name:
        Hugging Face / Sentence Transformers model name.

    batch_size:
        Number of chunks embedded in one batch.

    device:
        Device used by Sentence Transformers.
        "cpu" is the safe default for the current development
        environment.
    """

    DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
    DEFAULT_BATCH_SIZE = 16
    EXPECTED_DIMENSION = 768

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        # Dependency injection allows unit tests to use a fake model
        # without downloading or loading BGE-base.
        self._model = model

    @property
    def model(self) -> Any:
        """
        Lazily load the Sentence Transformer model.

        Lazy loading prevents model initialization when the object is
        created but embeddings are not yet requested.
        """

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading embedding model: {self.model_name}"
            )

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            logger.info(
                f"Embedding model loaded successfully: "
                f"{self.model_name}"
            )

        return self._model

    def embed(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for document chunks.

        Chunks are processed in batches to control memory usage.
        """

        if not chunks:
            logger.info("Embedder received no chunks.")
            return []

        texts = [chunk.text for chunk in chunks]

        logger.info(
            f"Generating embeddings for {len(texts)} chunks "
            f"using batch size {self.batch_size}."
        )

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding model returned a different number "
                "of embeddings than input chunks."
            )

        embedded_chunks: list[EmbeddedChunk] = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            vector = [float(value) for value in embedding]

            self._validate_embedding(vector)

            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    eku_id=chunk.eku_id,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    metadata=chunk.metadata,
                    embedding=vector,
                )
            )

        logger.info(
            f"Successfully generated {len(embedded_chunks)} embeddings."
        )

        return embedded_chunks

    def _validate_embedding(
        self,
        embedding: list[float],
    ) -> None:
        """
        Validate the generated embedding vector.
        """

        if not embedding:
            raise ValueError("Embedding vector cannot be empty.")

        if len(embedding) != self.EXPECTED_DIMENSION:
            raise ValueError(
                "Unexpected embedding dimension: "
                f"expected {self.EXPECTED_DIMENSION}, "
                f"got {len(embedding)}."
            )