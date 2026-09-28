"""
Core domain models for the CA Mentor AI ingestion pipeline.

These models represent the educational meaning of extracted
content after it has crossed the raw-document boundary.

Pipeline:

RawBlock
    ↓
EducationalElement
    ↓
EducationalKnowledgeUnit
    ↓
DocumentChunk
    ↓
EmbeddedChunk
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ============================================================
# DOCUMENT / ACADEMIC HIERARCHY
# ============================================================


class DocumentLevel(str, Enum):
    """CA academic level."""

    FOUNDATION = "foundation"
    INTERMEDIATE = "intermediate"
    FINAL = "final"


class SectionLevel(str, Enum):
    """Logical hierarchy inside a CA document."""

    PAPER = "paper"
    MODULE = "module"
    CHAPTER = "chapter"
    UNIT = "unit"
    TOPIC = "topic"
    SUBTOPIC = "subtopic"


# ============================================================
# EDUCATIONAL TYPES
# ============================================================


class EducationalType(str, Enum):
    """Educational meaning assigned to extracted content."""

    CONCEPT = "concept"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    FORMULA = "formula"
    WORKED_EXAMPLE = "worked_example"
    ILLUSTRATION = "illustration"

    PRACTICE_QUESTION = "practice_question"
    EXAM_QUESTION = "exam_question"

    MEMORY_TRICK = "memory_trick"
    SUMMARY = "summary"

    TABLE = "table"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"

    CASE_STUDY = "case_study"

    EXCEPTION = "exception"
    RULE = "rule"
    IMPORTANT_POINT = "important_point"

    LEARNING_OUTCOME = "learning_outcome"


class DifficultyLevel(str, Enum):
    """Estimated difficulty of the educational content."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class TeachingPriority(str, Enum):
    """Importance of content for teaching/retrieval."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================
# EMBEDDING STRATEGY
# ============================================================


class EmbeddingStrategy(str, Enum):
    """
    Determines whether and how an educational unit should
    participate in vector indexing.
    """

    NONE = "none"
    FULL = "full"
    SUMMARY_ONLY = "summary_only"
    TABLE_ONLY = "table_only"
    IMAGE_ONLY = "image_only"


# ============================================================
# RELATIONSHIPS
# ============================================================


class RelationshipType(str, Enum):
    """Relationship between educational knowledge units."""

    RELATED = "related"
    PREREQUISITE = "prerequisite"
    DEPENDS_ON = "depends_on"
    EXAMPLE_OF = "example_of"
    CONTINUATION_OF = "continuation_of"


# ============================================================
# DOCUMENT METADATA
# ============================================================


class DocumentMetadata(BaseModel):
    """Metadata describing the source document."""

    document_id: str
    filename: str
    title: str
    source: str
    edition: str
    language: str = "English"

    level: DocumentLevel
    paper: str

    total_pages: int = Field(default=0, ge=0)


# ============================================================
# RELATIONSHIP METADATA
# ============================================================


class RelationshipMetadata(BaseModel):
    """Relationship between the current EKU and another EKU."""

    relationship: RelationshipType
    target_id: str


# ============================================================
# CHUNK METADATA
# ============================================================


class ChunkMetadata(BaseModel):
    """
    Metadata attached to every retrieval chunk.

    This metadata will later be used by:
        - metadata filtering
        - BM25 retrieval
        - dense retrieval
        - reranking
        - citation generation
        - analytics
    """

    level: DocumentLevel
    paper: str

    module: str | None = None
    chapter: str | None = None
    unit: str | None = None
    topic: str | None = None
    subtopic: str | None = None

    educational_type: EducationalType

    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    teaching_priority: TeachingPriority = TeachingPriority.HIGH

    embedding_strategy: EmbeddingStrategy = EmbeddingStrategy.FULL

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    has_table: bool = False
    has_image: bool = False
    has_formula: bool = False
    has_example: bool = False

    version: str


# ============================================================
# TABLE / IMAGE METADATA
# ============================================================


class TableMetadata(BaseModel):
    """Metadata describing a table contained in an EKU."""

    title: str | None = None
    rows: int = Field(default=0, ge=0)
    columns: int = Field(default=0, ge=0)


class ImageMetadata(BaseModel):
    """Metadata describing an image/diagram."""

    caption: str | None = None
    image_type: str | None = None


# ============================================================
# EDUCATIONAL ELEMENT
# ============================================================


class EducationalElement(BaseModel):
    """
    Classified educational block.

    This is the direct output of the classifier.

    Example:

        RawBlock
            ↓
        EducationalElement(
            educational_type=CONCEPT,
            content="Accounting is...",
            hierarchy_path=[...],
            source_block_ids=[...]
        )
    """

    id: str

    educational_type: EducationalType

    content: str

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    hierarchy_path: list[str] = Field(default_factory=list)

    # Provenance back to RawBlock objects.
    source_block_ids: list[str] = Field(default_factory=list)

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # Additional classifier metadata.
    metadata: dict[str, str] = Field(default_factory=dict)


# ============================================================
# EDUCATIONAL KNOWLEDGE UNIT
# ============================================================


class EducationalKnowledgeUnit(BaseModel):
    """
    Semantically meaningful educational unit.

    The parser creates EKUs by combining one or more
    EducationalElements.

    Example:

        EducationalElement 1
        EducationalElement 2
        EducationalElement 3
                ↓
        EducationalKnowledgeUnit
    """

    id: str

    title: str

    educational_type: EducationalType

    content: str

    metadata: ChunkMetadata

    hierarchy_path: list[str] = Field(default_factory=list)

    # Provenance from EKU back to classifier output.
    source_element_ids: list[str] = Field(default_factory=list)

    relationships: list[RelationshipMetadata] = Field(
        default_factory=list
    )

    tables: list[TableMetadata] = Field(
        default_factory=list
    )

    images: list[ImageMetadata] = Field(
        default_factory=list
    )


# ============================================================
# CHUNKING
# ============================================================


class ChunkCandidate(BaseModel):
    """
    Candidate text produced before final chunk creation.
    """

    eku_id: str
    text: str
    metadata: ChunkMetadata


class DocumentChunk(BaseModel):
    """
    Retrieval-ready text chunk before embedding.

    One EKU may produce multiple DocumentChunks.
    """

    chunk_id: str

    # Parent Educational Knowledge Unit.
    eku_id: str

    # Position within the parent EKU.
    chunk_index: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=1, ge=1)

    # Retrieval text.
    text: str

    # Estimated token count in V1.
    token_count: int = Field(default=0, ge=0)

    metadata: ChunkMetadata

# ============================================================
# EMBEDDING
# ============================================================


class EmbeddedChunk(DocumentChunk):
    """
    Document chunk after embedding generation.
    """

    embedding: list[float]


# ============================================================
# INGESTION REPORT
# ============================================================


class IngestionReport(BaseModel):
    """
    Summary of one ingestion operation.
    """

    filename: str

    total_pages: int

    sections: int

    eku_count: int

    chunks: int

    tables: int

    images: int

    embedding_time: float

    upload_time: float

    warnings: list[str] = Field(
        default_factory=list
    )

    success: bool = True