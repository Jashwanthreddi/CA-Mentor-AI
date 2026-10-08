"""
Tests for the end-to-end ingestion pipeline orchestration.

These tests verify that IngestionPipeline:

    - calls every ingestion stage in the correct order
    - passes the output of one stage to the next
    - builds authoritative document metadata
    - passes document metadata into the parser
    - stops when validation fails
    - returns the uploader's IngestionReport
    - supports dependency injection

The tests intentionally do NOT:

    - process a real PDF
    - load the BGE model
    - connect to Qdrant Cloud
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.ingestion.models import (
    ChunkMetadata,
    DocumentChunk,
    EmbeddedChunk,
    DocumentMetadata,
    EducationalKnowledgeUnit,
    EducationalElement,
    EducationalType,
    IngestionReport,
)
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.raw_document import (
    RawDocument,
    RawDocumentMetadata,
    RawPage,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def create_raw_document() -> RawDocument:
    """Create a minimal RawDocument for pipeline testing."""

    return RawDocument(
        metadata=RawDocumentMetadata(
            filename="test.pdf",
            title="Test Document",
            pages=1,
            language="English",
            parser="test-parser",
            parser_version="test-v1",
        ),
        pages=[
            RawPage(
                page_number=1,
                width=600.0,
                height=800.0,
                blocks=[],
            )
        ],
    )


def create_document_metadata() -> DocumentMetadata:
    """Create authoritative document metadata for pipeline testing."""

    return DocumentMetadata(
        document_id="doc-test-1",
        filename="test.pdf",
        title="Test Document",
        source="ICAI",
        edition="test-edition",
        language="English",
        level="foundation",
        paper="Paper 1",
        total_pages=1,
    )


def create_element() -> EducationalElement:
    """Create a minimal EducationalElement."""

    return EducationalElement(
        id="element-1",
        educational_type=EducationalType.CONCEPT,
        content="Accounting is the process of recording financial transactions.",
        page_start=1,
        page_end=1,
        hierarchy_path=["Chapter 1", "Unit 1"],
        source_block_ids=["block-1"],
    )


def create_knowledge_unit() -> EducationalKnowledgeUnit:
    """Create a minimal EducationalKnowledgeUnit."""

    metadata = ChunkMetadata(
        document_id="doc-test-1",
        source="ICAI",
        edition="test-edition",
        level="foundation",
        paper="Paper 1",
        module="Module 1",
        chapter="Chapter 1",
        unit="Unit 1",
        topic="Introduction",
        educational_type=EducationalType.CONCEPT,
        page_start=1,
        page_end=1,
        version="metadata-v1",
    )

    return EducationalKnowledgeUnit(
        id="eku-1",
        title="Introduction",
        educational_type=EducationalType.CONCEPT,
        content="Accounting is the process of recording financial transactions.",
        metadata=metadata,
        hierarchy_path=["Chapter 1", "Unit 1"],
        source_element_ids=["element-1"],
    )


def create_chunk() -> DocumentChunk:
    """Create a minimal DocumentChunk."""

    metadata = ChunkMetadata(
        document_id="doc-test-1",
        source="ICAI",
        edition="test-edition",
        level="foundation",
        paper="Paper 1",
        module="Module 1",
        chapter="Chapter 1",
        unit="Unit 1",
        topic="Introduction",
        educational_type=EducationalType.CONCEPT,
        page_start=1,
        page_end=1,
        version="metadata-v1",
    )

    return DocumentChunk(
        chunk_id="chunk-1",
        eku_id="eku-1",
        chunk_index=0,
        total_chunks=1,
        text="Accounting is the process of recording financial transactions.",
        token_count=10,
        metadata=metadata,
    )


def create_embedded_chunk() -> EmbeddedChunk:
    """Create a minimal EmbeddedChunk."""

    metadata = ChunkMetadata(
        document_id="doc-test-1",
        source="ICAI",
        edition="test-edition",
        level="foundation",
        paper="Paper 1",
        module="Module 1",
        chapter="Chapter 1",
        unit="Unit 1",
        topic="Introduction",
        educational_type=EducationalType.CONCEPT,
        page_start=1,
        page_end=1,
        version="metadata-v1",
    )

    return EmbeddedChunk(
        chunk_id="chunk-1",
        eku_id="eku-1",
        chunk_index=0,
        total_chunks=1,
        text="Accounting is the process of recording financial transactions.",
        token_count=10,
        metadata=metadata,
        embedding=[0.1] * 768,
    )


def create_ingestion_report() -> IngestionReport:
    """Create a minimal successful ingestion report."""

    return IngestionReport(
        filename="test.pdf",
        total_pages=1,
        sections=1,
        eku_count=1,
        chunks=1,
        tables=0,
        images=0,
        embedding_time=0.1,
        upload_time=0.2,
        warnings=[],
        success=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_components():
    """
    Create mocked pipeline components.

    Dependency injection allows us to test orchestration without
    running real PDF extraction, BGE inference, or Qdrant.
    """

    loader = MagicMock()
    extractor = MagicMock()
    document_metadata_builder = MagicMock()
    classifier = MagicMock()
    parser = MagicMock()
    chunker = MagicMock()
    metadata_builder = MagicMock()
    validator = MagicMock()
    embedder = MagicMock()
    uploader = MagicMock()

    raw_document = create_raw_document()
    document_metadata = create_document_metadata()
    elements = [create_element()]
    knowledge_units = [create_knowledge_unit()]
    chunks = [create_chunk()]
    embedded_chunks = [create_embedded_chunk()]
    report = create_ingestion_report()

    loader.load.return_value = Path("test.pdf")

    extractor.extract.return_value = raw_document

    document_metadata_builder.build.return_value = document_metadata

    classifier.classify.return_value = elements

    parser.parse.return_value = knowledge_units

    chunker.chunk.return_value = chunks

    metadata_builder.build.return_value = chunks

    validation_result = MagicMock()
    validation_result.is_valid = True
    validation_result.valid_count = 1
    validation_result.invalid_count = 0
    validation_result.valid_chunks = chunks
    validation_result.errors = []

    validator.validate.return_value = validation_result

    embedder.embed.return_value = embedded_chunks

    uploader.upload.return_value = report

    pipeline = IngestionPipeline(
        loader=loader,
        extractor=extractor,
        document_metadata_builder=document_metadata_builder,
        classifier=classifier,
        parser=parser,
        chunker=chunker,
        metadata_builder=metadata_builder,
        validator=validator,
        embedder=embedder,
        uploader=uploader,
    )

    return {
        "pipeline": pipeline,
        "loader": loader,
        "extractor": extractor,
        "document_metadata_builder": document_metadata_builder,
        "classifier": classifier,
        "parser": parser,
        "chunker": chunker,
        "metadata_builder": metadata_builder,
        "validator": validator,
        "embedder": embedder,
        "uploader": uploader,
        "raw_document": raw_document,
        "document_metadata": document_metadata,
        "elements": elements,
        "knowledge_units": knowledge_units,
        "chunks": chunks,
        "embedded_chunks": embedded_chunks,
        "report": report,
        "validation_result": validation_result,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_runs_successfully(pipeline_components):
    """The complete pipeline should execute successfully."""

    pipeline = pipeline_components["pipeline"]

    result = pipeline.run(
        Path("test.pdf"),
        page_range=(1, 3),
    )

    assert result.success is True
    assert result.filename == "test.pdf"
    assert result.chunks == 1


def test_pipeline_calls_all_stages(pipeline_components):
    """Every ingestion stage should be called exactly once."""

    pipeline = pipeline_components["pipeline"]

    pipeline.run(
        Path("test.pdf"),
        page_range=(1, 3),
    )

    pipeline_components["loader"].load.assert_called_once()
    pipeline_components["extractor"].extract.assert_called_once()
    pipeline_components["document_metadata_builder"].build.assert_called_once()
    pipeline_components["classifier"].classify.assert_called_once()
    pipeline_components["parser"].parse.assert_called_once()
    pipeline_components["chunker"].chunk.assert_called_once()
    pipeline_components["metadata_builder"].build.assert_called_once()
    pipeline_components["validator"].validate.assert_called_once()
    pipeline_components["embedder"].embed.assert_called_once()
    pipeline_components["uploader"].upload.assert_called_once()


def test_pipeline_passes_page_range_to_extractor(pipeline_components):
    """The requested page range should reach the extractor."""

    pipeline = pipeline_components["pipeline"]

    pipeline.run(
        Path("test.pdf"),
        page_range=(1, 3),
    )

    pipeline_components["extractor"].extract.assert_called_once_with(
        Path("test.pdf"),
        page_range=(1, 3),
    )


def test_pipeline_preserves_stage_data_flow(pipeline_components):
    """
    Each stage should receive the output of the previous stage.
    """

    pipeline = pipeline_components["pipeline"]

    pipeline.run(Path("test.pdf"))

    loader = pipeline_components["loader"]
    extractor = pipeline_components["extractor"]
    document_metadata_builder = pipeline_components["document_metadata_builder"]
    classifier = pipeline_components["classifier"]
    parser = pipeline_components["parser"]
    chunker = pipeline_components["chunker"]
    metadata_builder = pipeline_components["metadata_builder"]
    validator = pipeline_components["validator"]
    embedder = pipeline_components["embedder"]
    uploader = pipeline_components["uploader"]

    extractor.extract.assert_called_once_with(
        Path("test.pdf"),
        page_range=None,
    )

    document_metadata_builder.build.assert_called_once_with(
        pdf_path=Path("test.pdf"),
        raw_document=pipeline_components["raw_document"],
    )

    classifier.classify.assert_called_once_with(
        pipeline_components["raw_document"]
    )

    parser.parse.assert_called_once_with(
        pipeline_components["elements"],
        document_metadata=pipeline_components["document_metadata"],
    )

    chunker.chunk.assert_called_once_with(
        pipeline_components["knowledge_units"]
    )

    metadata_builder.build.assert_called_once_with(
        pipeline_components["chunks"]
    )

    validator.validate.assert_called_once_with(
        pipeline_components["chunks"]
    )

    embedder.embed.assert_called_once_with(
        pipeline_components["chunks"]
    )
    uploader.upload.assert_called_once()
    # uploader.assert_called_once() if False else None


def test_pipeline_returns_uploader_report(pipeline_components):
    """The pipeline should return the uploader's report."""

    pipeline = pipeline_components["pipeline"]

    result = pipeline.run(Path("test.pdf"))

    assert result is pipeline_components["report"]


def test_pipeline_uploads_embedded_chunks(pipeline_components):
    """The uploader should receive the embedded chunks."""

    pipeline = pipeline_components["pipeline"]

    pipeline.run(Path("test.pdf"))

    uploader = pipeline_components["uploader"]

    call_args = uploader.upload.call_args

    assert call_args is not None

    embedded_chunks = call_args.args[0]

    assert embedded_chunks == pipeline_components["embedded_chunks"]


def test_pipeline_passes_document_statistics_to_uploader(
    pipeline_components,
):
    """Document-level statistics should be passed to the uploader."""

    pipeline = pipeline_components["pipeline"]

    pipeline.run(Path("test.pdf"))

    uploader = pipeline_components["uploader"]

    kwargs = uploader.upload.call_args.kwargs

    assert kwargs["filename"] == "test.pdf"
    assert kwargs["total_pages"] == 1
    assert kwargs["sections"] == 1
    assert kwargs["eku_count"] == 1
    assert kwargs["tables"] == 0
    assert kwargs["images"] == 0
    assert kwargs["embedding_time"] >= 0


def test_pipeline_stops_when_validation_fails(pipeline_components):
    """
    Invalid chunks should stop the pipeline before embedding/upload.
    """

    pipeline = pipeline_components["pipeline"]

    validation_result = pipeline_components["validation_result"]

    validation_result.is_valid = False
    validation_result.valid_count = 0
    validation_result.invalid_count = 1
    validation_result.valid_chunks = []
    validation_result.errors = [
        "chunk-1: invalid page provenance"
    ]

    with pytest.raises(ValueError, match="Ingestion validation failed"):
        pipeline.run(Path("test.pdf"))

    pipeline_components["embedder"].embed.assert_not_called()
    pipeline_components["uploader"].upload.assert_not_called()


def test_pipeline_uses_valid_chunks_for_embedding(
    pipeline_components,
):
    """Only validated chunks should be sent to the embedder."""

    pipeline = pipeline_components["pipeline"]

    pipeline.run(Path("test.pdf"))

    pipeline_components["embedder"].embed.assert_called_once_with(
        pipeline_components["chunks"]
    )


def test_pipeline_supports_dependency_injection():
    """All pipeline components should be injectable."""

    components = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]

    pipeline = IngestionPipeline(
        loader=components[0],
        extractor=components[1],
        document_metadata_builder=components[2],
        classifier=components[3],
        parser=components[4],
        chunker=components[5],
        metadata_builder=components[6],
        validator=components[7],
        embedder=components[8],
        uploader=components[9],
    )

    assert pipeline.loader is components[0]
    assert pipeline.extractor is components[1]
    assert pipeline.document_metadata_builder is components[2]
    assert pipeline.classifier is components[3]
    assert pipeline.parser is components[4]
    assert pipeline.chunker is components[5]
    assert pipeline.metadata_builder is components[6]
    assert pipeline.validator is components[7]
    assert pipeline.embedder is components[8]
    assert pipeline.uploader is components[9]


def test_pipeline_calculates_table_count(pipeline_components):
    """Chunks containing tables should contribute to table statistics."""

    chunk = pipeline_components["chunks"][0]
    chunk.metadata.has_table = True

    pipeline = pipeline_components["pipeline"]

    pipeline.run(Path("test.pdf"))

    kwargs = pipeline_components["uploader"].upload.call_args.kwargs

    assert kwargs["tables"] == 1


def test_pipeline_calculates_image_count(pipeline_components):
    """Chunks containing images should contribute to image statistics."""

    chunk = pipeline_components["chunks"][0]
    chunk.metadata.has_image = True

    pipeline = pipeline_components["pipeline"]

    pipeline.run(Path("test.pdf"))

    kwargs = pipeline_components["uploader"].upload.call_args.kwargs

    assert kwargs["images"] == 1