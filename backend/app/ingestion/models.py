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
from __future__ import annotations # allow forward references

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# enums

class DocumentLevel(str,Enum):
    FOUNDATION="foundation"
    INTERMEDIATE="intermediate"
    FINAL="final"

class SectionLevel(str,Enum):
    PAPER="paper"
    MODULE="module"
    CHAPTER="chapter"
    UNIT="unit"
    TOPIC="topic"
    SUBTOPIC="subtopic"

class EducationalType(str,Enum):
    CONCEPT="concept"
    DEFINITION="definition"
    PROCEDURE="procedure"
    FORMULA="formula"
    WORKED_EXAMPLE="worked_example"
    ILLUSTRATION="illustration"
    PRACTICE_QUESTION="practice_question"
    EXAM_QUESTION="exam_question"
    MEMORY_TRICK="memory_trick"
    SUMMARY="summary"
    TABLE="table"
    DIAGRAM="diagram"
    FLOWCHART="flowchart"
    CASE_STUDY="case_study"
    EXCEPTION="exception"
    RULE="rule"
    IMPORTANT_POINT="important_point"
    LEARNING_OUTCOME="learning_outcome"

class DifficultyLevel(str,Enum):
    BEGINNER="beginner"
    INTERMEDIATE="intermediate"
    ADVANCE="advance"

class TeachingPriority(str,Enum):
    HIGH="high"
    MEDIUM="medium"
    LOW="low"

class EmbeddingStrategy(str,Enum):
    None="none"
    FULL="full"
    SUMMARY_ONLY="summary_only"
    TABLE_ONLY="table_only"
    IMAGE_ONLY="image_only"

class RelationshipType(str,Enum):
    RELATED="related"
    PREREQUISITE="prerequisite"
    DEPENDS_ON="depends_on"
    EXAMPLE_OF="eample_of"
    CONTINUATION_OF="continuation_of"

# metadata models
class DocumentMetadata(BaseModel):
    #metadata describing the complete source document
    filename:str
    title:str
    source:str
    edition:str
    language:str="English"

    level:DocumentLevel
    paper:str

    total_pages:int=0


class RelationshipMetadata(BaseModel):
    # relationship together with every chunk

    relationship:RelationshipType
    target_id:str


class ChunkMetadata(BaseModel):
    # metadata stored together with every chunk
    level:DocumentLevel
    paper:str
    module:str|None =None
    chapter :str | None=None
    unit:str|None=None
    topic:str |None=None
    subtopic :str |None=None
    educational_type: EducationalType
    difficulty:DifficultyLevel=DifficultyLevel.BEGINNER
    teaching_priority:TeachingPriority=TeachingPriority.HIGH
    embedding_strategy: EmbeddingStrategy=EmbeddingStrategy.FULL
    page_start:int
    page_end:int
    has_table:bool=False
    has_image:bool=False
    has_formula:bool=False
    has_example:bool=False
    version:str


class TableMetadata(BaseModel):
    title:str  |None=None
    row:int=0
    columns:int=0


class ImageMetadata(BaseModel):
    caption:str |None=None
    image_type:str |None=None


# Core Document models

class PDFDocument(BaseModel):
    # represents one educational documents.
    metadata:DocumentMetadata
    sections:list["DocumentSection"]=Field(default_factory=list)

class DocumentSection(BaseModel):
    #represents document hierarchy
    id :str
    title:str
    level:SectionLevel
    page_start:int
    page_end:int
    content:str=""
    children:list["DocumentSection"]=Field(default_factory=list)

class EducationalKnowledgeUnit(BaseModel):
    # smallest complete educational object
    id:str
    title:str
    educational_type:EducationalType
    content:str
    metadata:ChunkMetadata
    hierarchy_path:list[str]
    relationships:list[RelationshipMetadata]=Field(default_factory=list)
    tables:list[TableMetadata]=Field(default_factory=list)
    images:list[ImageMetadata]=Field(default_factory=list)


class ChunkCandiate(BaseModel):
    # intermediate object before chunk generation
    eku_id:str
    text:str
    metadata:ChunkMetadata


class DocumentChunk(BaseModel):
    #smallest retrievable knowledge unit.
    chunk_id:str
    eku_id:str
    text:str
    metadata:ChunkMetadata


class EmbeddedChunk(DocumentChunk):
    # documnet chunk with embedding
    embedding:list[float]

    #--------------------------------
    # ingestion report
    #------------------------------


class IngestionReport(BaseModel):
    filename:str
    total_pages:int
    section:int
    eku_count:int
    chunks:int
    tables:int
    images:int
    embedding_time:float
    upload_time:float
    warnings:list[str] =Field(default_factory=list)
    success:bool=True

    