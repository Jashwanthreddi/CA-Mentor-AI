"""
Document metadata construction for the CA Mentor AI ingestion pipeline.

This module is responsible for creating authoritative academic/source
metadata for an ingested document.

It intentionally does NOT inspect educational content.

Responsibilities
----------------
- Build stable document identity.
- Extract academic hierarchy from the ingestion path.
- Preserve source/edition supplied by the caller.
- Build DocumentMetadata from RawDocument information.

It does NOT:
- classify content
- parse educational meaning
- inspect PDF blocks
- generate embeddings
- access Qdrant
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.ingestion.models import (
    DocumentLevel,
    DocumentMetadata,
)
from app.ingestion.raw_document import RawDocument


class DocumentMetadataBuilder:
    """
    Build authoritative DocumentMetadata for an ingestion job.

    The ingestion directory convention is expected to look like:

        .../
        foundation/
        paper_1/
        module_1/
        chapter_1/
        file.pdf

    The builder uses those directory names only as identifiers.

    It does not attempt to guess human-readable paper names.
    """

    DEFAULT_SOURCE = "ICAI"

    DEFAULT_EDITION = "unspecified"

    LEVEL_PATTERN = re.compile(
        r"^(foundation|intermediate|final)$",
        re.IGNORECASE,
    )

    PAPER_PATTERN = re.compile(
        r"^paper[_\-\s]*(.+)$",
        re.IGNORECASE,
    )

    @classmethod
    def build(
        cls,
        *,
        pdf_path: Path,
        raw_document: RawDocument,
        source: str | None = None,
        edition: str | None = None,
    ) -> DocumentMetadata:
        """
        Build DocumentMetadata from the source path and raw document.

        Parameters
        ----------
        pdf_path:
            Original PDF path.

        raw_document:
            Structural document returned by the extractor.

        source:
            Authoritative source name.

        edition:
            Authoritative study-material edition/reprint identifier.
        """

        level = cls._extract_level(pdf_path)

        paper = cls._extract_paper(pdf_path)

        document_id = cls._generate_document_id(
            pdf_path=pdf_path,
            level=level,
            paper=paper,
        )

        return DocumentMetadata(
            document_id=document_id,
            filename=raw_document.metadata.filename,
            title=raw_document.metadata.title,
            source=(
                source.strip()
                if source and source.strip()
                else cls.DEFAULT_SOURCE
            ),
            edition=(
                edition.strip()
                if edition and edition.strip()
                else cls.DEFAULT_EDITION
            ),
            language=raw_document.metadata.language,
            level=level,
            paper=paper,
            total_pages=len(raw_document.pages),
        )

    # ========================================================
    # PATH PARSING
    # ========================================================

    @classmethod
    def _extract_level(
        cls,
        pdf_path: Path,
    ) -> DocumentLevel:
        """
        Extract CA level from the path.

        Example:

            foundation/paper_1/...
        """

        for part in reversed(pdf_path.parts):
            match = cls.LEVEL_PATTERN.match(
                part.strip()
            )

            if not match:
                continue

            return DocumentLevel(
                match.group(1).lower()
            )

        raise ValueError(
            "Unable to determine CA document level from path. "
            "Expected a path containing foundation, intermediate, "
            "or final."
        )

    @classmethod
    def _extract_paper(
        cls,
        pdf_path: Path,
    ) -> str:
        """
        Extract the paper identifier from the path.

        Example:

            paper_1 → paper_1

        We deliberately keep this as an identifier rather than
        guessing a human-readable paper title.
        """

        for part in reversed(pdf_path.parts):
            normalized = part.strip()

            match = cls.PAPER_PATTERN.match(
                normalized
            )

            if match:
                suffix = match.group(1).strip()

                if suffix:
                    return f"paper_{suffix}"

        raise ValueError(
            "Unable to determine paper from path. "
            "Expected a directory such as paper_1."
        )

    # ========================================================
    # DOCUMENT ID
    # ========================================================

    @staticmethod
    def _generate_document_id(
        *,
        pdf_path: Path,
        level: DocumentLevel,
        paper: str,
    ) -> str:
        """
        Generate a stable document identifier.

        The absolute path is normalized before hashing so repeated
        ingestion of the same source produces the same ID.
        """

        identity = (
            f"{level.value}|"
            f"{paper}|"
            f"{pdf_path.resolve().as_posix().lower()}"
        )

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:20]

        return f"doc-{digest}"