"""
Metadata Builder

Purpose
-------
Normalize and finalize ChunkMetadata for parsed document chunks.

The builder operates AFTER parsing and chunking.

Responsibilities
----------------
- Preserve document identity.
- Preserve academic hierarchy.
- Preserve educational type.
- Preserve page provenance.
- Preserve structural flags.
- Preserve difficulty and teaching priority.
- Preserve embedding strategy.
- Set the metadata schema version.

The builder does NOT:
- classify content
- infer document identity
- reconstruct hierarchy
- generate embeddings
- access Qdrant
- access external services
"""

from __future__ import annotations

from app.core.logging import logger
from app.ingestion.models import DocumentChunk


class MetadataBuilder:
    """
    Build final retrieval/provenance metadata for document chunks.

    The builder is deterministic and performs no external I/O.
    """

    METADATA_VERSION = "metadata-v2"

    def build(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """
        Finalize metadata for all document chunks.
        """

        if not chunks:
            logger.info(
                "Metadata builder received no chunks."
            )

            return []

        enriched_chunks = [
            self._enrich_chunk(chunk)
            for chunk in chunks
        ]

        logger.info(
            "Metadata enrichment completed for "
            f"{len(enriched_chunks)} chunks."
        )

        return enriched_chunks

    def _enrich_chunk(
        self,
        chunk: DocumentChunk,
    ) -> DocumentChunk:
        """
        Finalize metadata for one chunk.

        Existing semantic metadata is preserved.

        Only normalization/version/provenance fields are
        modified here.
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