"""
Metadata Builder

Purpose
-------
Build and enrich ChunkMetadata for parsed document chunks.

The metadata builder converts information already present in the
DocumentChunk and its source EKU into retrieval-friendly metadata.

Responsibilities
----------------
- Preserve educational hierarchy.
- Preserve educational type.
- Preserve page provenance.
- Preserve structural flags.
- Preserve difficulty and teaching priority.
- Preserve embedding strategy.
- Add the metadata version.

The builder does NOT:
- classify content
- parse educational meaning
- generate embeddings
- access Qdrant
- access external services
"""

from __future__ import annotations

from app.core.logging import logger
from app.ingestion.models import DocumentChunk


class MetadataBuilder:
    """
    Build retrieval and provenance metadata for document chunks.

    The builder is deterministic and does not perform external I/O.
    """

    METADATA_VERSION = "metadata-v1"

    def build(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """
        Enrich all document chunks with metadata.

        Parameters
        ----------
        chunks:
            Document chunks produced by the chunking stage.

        Returns
        -------
        list[DocumentChunk]
            Document chunks with enriched metadata.
        """

        if not chunks:
            logger.info("Metadata builder received no chunks.")
            return []

        enriched_chunks = [
            self._enrich_chunk(chunk)
            for chunk in chunks
        ]

        logger.info(
            f"Metadata enrichment completed for "
            f"{len(enriched_chunks)} chunks."
        )

        return enriched_chunks

    def _enrich_chunk(
        self,
        chunk: DocumentChunk,
    ) -> DocumentChunk:
        """
        Enrich metadata for a single document chunk.

        Existing metadata values are preserved. The chunk-level
        provenance fields are synchronized with the DocumentChunk.
        """

        metadata = chunk.metadata.model_copy(
            update={
                "page_start": chunk.metadata.page_start,
                "page_end": chunk.metadata.page_end,
                "version": self.METADATA_VERSION,
            }
        )

        return chunk.model_copy(
            update={
                "metadata": metadata,
            }
        )