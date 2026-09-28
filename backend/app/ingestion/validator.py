"""
Document Chunk Validator

Purpose
-------
Validate DocumentChunk objects before they are sent to the
embedding and vector database stages.

The validator protects the ingestion pipeline from invalid
retrieval data while allowing valid chunks to continue.

Responsibilities
----------------
- Validate chunk text.
- Validate chunk identity.
- Validate chunk ordering.
- Validate token counts.
- Validate page provenance.
- Validate metadata consistency.
- Detect duplicate chunk IDs.
- Return valid chunks without mutating them.

The validator does NOT:
- modify chunk content
- create metadata
- generate embeddings
- access Qdrant
- access external services
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import logger
from app.ingestion.models import DocumentChunk


@dataclass
class ValidationResult:
    """
    Result produced by the document chunk validator.
    """

    valid_chunks: list[DocumentChunk] = field(default_factory=list)
    invalid_chunks: list[DocumentChunk] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        """Return the total number of chunks processed."""
        return len(self.valid_chunks) + len(self.invalid_chunks)

    @property
    def valid_count(self) -> int:
        """Return the number of valid chunks."""
        return len(self.valid_chunks)

    @property
    def invalid_count(self) -> int:
        """Return the number of invalid chunks."""
        return len(self.invalid_chunks)

    @property
    def is_valid(self) -> bool:
        """Return True when all chunks are valid."""
        return self.invalid_count == 0


class DocumentChunkValidator:
    """
    Validate document chunks before embedding.

    Validation is deterministic and side-effect free.
    """

    def validate(
        self,
        chunks: list[DocumentChunk],
    ) -> ValidationResult:
        """
        Validate a collection of document chunks.

        Invalid chunks are recorded instead of immediately raising
        an exception so that the rest of the valid document can
        continue through the ingestion pipeline.
        """

        if not chunks:
            logger.info("Validator received no chunks.")
            return ValidationResult()

        result = ValidationResult()

        seen_chunk_ids: set[str] = set()

        for chunk in chunks:
            errors = self._validate_chunk(
                chunk=chunk,
                seen_chunk_ids=seen_chunk_ids,
            )

            if errors:
                result.invalid_chunks.append(chunk)
                result.errors.extend(
                    f"{chunk.chunk_id}: {error}"
                    for error in errors
                )
            else:
                result.valid_chunks.append(chunk)

            seen_chunk_ids.add(chunk.chunk_id)

        logger.info(
            f"Validation completed: "
            f"{result.valid_count} valid, "
            f"{result.invalid_count} invalid, "
            f"{result.total_chunks} total."
        )

        return result

    def _validate_chunk(
        self,
        *,
        chunk: DocumentChunk,
        seen_chunk_ids: set[str],
    ) -> list[str]:
        """
        Validate a single DocumentChunk.

        Returns a list of validation errors.
        An empty list means the chunk is valid.
        """

        errors: list[str] = []

        # ---------------------------------------------------------
        # Chunk identity
        # ---------------------------------------------------------

        if not chunk.chunk_id.strip():
            errors.append("chunk_id cannot be empty.")

        if chunk.chunk_id in seen_chunk_ids:
            errors.append(
                f"duplicate chunk_id detected: {chunk.chunk_id}"
            )

        if not chunk.eku_id.strip():
            errors.append("eku_id cannot be empty.")

        # ---------------------------------------------------------
        # Chunk content
        # ---------------------------------------------------------

        if not chunk.text.strip():
            errors.append("chunk text cannot be empty.")

        # ---------------------------------------------------------
        # Chunk ordering
        # ---------------------------------------------------------

        if chunk.chunk_index < 0:
            errors.append("chunk_index cannot be negative.")

        if chunk.total_chunks < 1:
            errors.append("total_chunks must be at least 1.")

        if chunk.chunk_index >= chunk.total_chunks:
            errors.append(
                "chunk_index must be smaller than total_chunks."
            )

        # ---------------------------------------------------------
        # Token count
        # ---------------------------------------------------------

        if chunk.token_count < 0:
            errors.append("token_count cannot be negative.")

        # ---------------------------------------------------------
        # Page provenance
        # ---------------------------------------------------------

        if chunk.metadata.page_start < 1:
            errors.append("page_start must be at least 1.")

        if chunk.metadata.page_end < 1:
            errors.append("page_end must be at least 1.")

        if chunk.metadata.page_start > chunk.metadata.page_end:
            errors.append(
                "page_start cannot be greater than page_end."
            )

        # ---------------------------------------------------------
        # Metadata consistency
        # ---------------------------------------------------------

        if chunk.metadata.version.strip() == "":
            errors.append("metadata version cannot be empty.")

        if chunk.metadata.paper.strip() == "":
            errors.append("paper cannot be empty.")

        return errors