#define lang spoken  by ingestio pipeline
# every stage will exchanfe typed objects instead of random dictonaries.
# respo: represent parsed docs, represent sections, rep chunks, rep embedded chunks, validate data.
# used by : loader.py -> parser.py -> chunker.py -> embedder.py -> uploader.py

"""
Knowledge Engineering Models

These models define the complete data contract for the Knowledge
Engineering Pipeline.

Pipeline

PDF
    ↓
Document
    ↓
Sections
    ↓
Educational Knowledge Units (EKUs)
    ↓
Chunk Candidates
    ↓
Chunks
    ↓
Embeddings
"""
"""
Core domain models for the CA Mentor AI ingestion pipeline.

These models represent the educational meaning of extracted
content after it has crossed the raw-document boundary.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DocumentLevel(str, Enum):
    """Academic level of the source document."""

    FOUNDATION = "foundation"
    INTERMEDIATE = "intermediate"
    FINAL = "final"


class SectionLevel(str, Enum):
    """Hierarchical level of the source material."""

    PAPER = "paper"
    MODULE = "module"
    CHAPTER = "chapter"
    UNIT = "unit"
    TOPIC = "topic"
    SUBTOPIC = "subtopic"


class EducationalType(str, Enum):
    """Educational role of a knowledge unit."""

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
    """Estimated learning difficulty."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class TeachingPriority(str, Enum):
    """Priority of the content for teaching and retrieval."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EmbeddingStrategy(str, Enum):
    """Controls how a knowledge unit is represented for retrieval."""

    NONE = "none"
    FULL = "full"
    SUMMARY_ONLY = "summary_only"
    TABLE_ONLY = "table_only"
    IMAGE_ONLY = "image_only"


class RelationshipType(str, Enum):
    """Relationship between educational knowledge units."""

    RELATED = "related"
    PREREQUISITE = "prerequisite"
    DEPENDS_ON = "depends_on"
    EXAMPLE_OF = "example_of"
    CONTINUATION_OF = "continuation_of"


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

    total_pages: int = 0


class RelationshipMetadata(BaseModel):
    """Relationship between two educational knowledge units."""

    relationship: RelationshipType
    target_id: str


class ChunkMetadata(BaseModel):
    """Metadata attached to retrieval chunks."""

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

    page_start: int
    page_end: int

    has_table: bool = False
    has_image: bool = False
    has_formula: bool = False
    has_example: bool = False

    version: str


class TableMetadata(BaseModel):
    """Metadata describing a table associated with a knowledge unit."""

    title: str | None = None
    rows: int = 0
    columns: int = 0


class ImageMetadata(BaseModel):
    """Metadata describing an image or diagram."""

    caption: str | None = None
    image_type: str | None = None


class EducationalElement(BaseModel):
    """
    Classified educational element produced by the classifier.

    This is an intermediate representation between the raw PDF
    representation and the final educational knowledge unit.
    """

    id: str

    educational_type: EducationalType

    content: str

    page_start: int
    page_end: int

    hierarchy_path: list[str] = Field(default_factory=list)

    source_block_ids: list[str] = Field(default_factory=list)

    confidence: float | None = None

    metadata: dict[str, str] = Field(default_factory=dict)


class EducationalKnowledgeUnit(BaseModel):
    """
    Semantically meaningful educational unit.

    A knowledge unit should represent a coherent piece of
    educational information rather than an arbitrary text window.
    """

    id: str

    title: str

    educational_type: EducationalType

    content: str

    metadata: ChunkMetadata

    hierarchy_path: list[str] = Field(default_factory=list)

    relationships: list[RelationshipMetadata] = Field(
        default_factory=list
    )

    tables: list[TableMetadata] = Field(
        default_factory=list
    )

    images: list[ImageMetadata] = Field(
        default_factory=list
    )


class ChunkCandidate(BaseModel):
    """Intermediate candidate before final chunk validation."""

    eku_id: str

    text: str

    metadata: ChunkMetadata


class DocumentChunk(BaseModel):
    """Final retrieval unit before embedding."""

    chunk_id: str

    eku_id: str

    text: str

    metadata: ChunkMetadata


class EmbeddedChunk(DocumentChunk):
    """Retrieval chunk with its embedding vector."""

    embedding: list[float]


class IngestionReport(BaseModel):
    """Summary of one ingestion execution."""

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