"""
Raw document models for the CA Mentor AI ingestion pipeline.

These models represent the structural content extracted from a source
document before educational classification and knowledge processing.

The models intentionally remain independent of Docling's internal
data structures so the rest of the application is not tightly coupled
to the extraction library.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    """Type of content block extracted from a document."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    HEADER = "header"
    FOOTER = "footer"


class BoundingBox(BaseModel):
    """
    Spatial coordinates of a document block.

    Coordinates are represented using a top-left-origin system:

        (left, top)
              ┌───────────────┐
              │               │
              │     block     │
              │               │
              └───────────────┘
                    (right, bottom)

    The extractor is responsible for converting the source parser's
    coordinate system into this normalized representation.
    """

    left: float
    top: float
    right: float
    bottom: float


class RawBlock(BaseModel):
    """
    A single structurally extracted block from a document page.

    A block represents the physical content discovered by the
    extraction layer. It does not contain educational interpretation.
    """

    id: str

    page: int = Field(
        ge=1,
        description="1-based physical PDF page number.",
    )

    block_type: BlockType

    text: str = ""

    bbox: BoundingBox | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional extraction confidence score.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parser-specific provenance metadata.",
    )


class RawPage(BaseModel):
    """
    Represents one physically extracted PDF page.

    Blocks preserve the document's reading order as returned by the
    extraction layer.
    """

    page_number: int = Field(
        ge=1,
        description="1-based physical PDF page number.",
    )

    width: float = Field(
        gt=0,
        description="Page width in source document units.",
    )

    height: float = Field(
        gt=0,
        description="Page height in source document units.",
    )

    blocks: list[RawBlock] = Field(
        default_factory=list,
    )


class RawDocumentMetadata(BaseModel):
    """
    Metadata describing the source document and extraction process.
    """

    filename: str

    title: str

    pages: int = Field(
        ge=0,
        description="Number of pages represented in the raw document.",
    )

    language: str = "English"

    parser: str

    parser_version: str


class RawDocument(BaseModel):
    """
    Complete structural representation of an extracted document.

    This is the output contract of the PDF extractor.

    Flow:

        PDF
         ↓
        Docling
         ↓
        RawDocument
         ↓
        Classifier
    """

    metadata: RawDocumentMetadata

    pages: list[RawPage] = Field(
        default_factory=list,
    )