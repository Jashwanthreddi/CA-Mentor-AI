from app.ingestion.metadata import MetadataBuilder
from app.ingestion.models import (
    ChunkMetadata,
    DocumentChunk,
    DocumentLevel,
    EducationalType,
)


def create_chunk(
    *,
    chunk_id: str = "chunk-1",
    eku_id: str = "eku-1",
    text: str = "Accounting is the process of recording transactions.",
) -> DocumentChunk:
    """
    Create a representative DocumentChunk for metadata tests.
    """

    metadata = ChunkMetadata(
    
        document_id="doc-test-foundation-paper1",
        source="ICAI",
        edition="test-edition",
        level=DocumentLevel.FOUNDATION,
        paper="Paper 1",
        module="Module 1",
        chapter="Chapter 1",
        unit="Unit 1",
        topic="Introduction",
        subtopic=None,
        educational_type=EducationalType.CONCEPT,
        page_start=2,
        page_end=3,
        has_table=False,
        has_image=False,
        has_formula=False,
        has_example=False,
        version="initial",
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        eku_id=eku_id,
        chunk_index=0,
        total_chunks=1,
        text=text,
        token_count=10,
        metadata=metadata,
    )


def test_empty_chunks_returns_empty_list():
    builder = MetadataBuilder()

    result = builder.build([])

    assert result == []


def test_metadata_builder_preserves_chunk_count():
    builder = MetadataBuilder()

    chunks = [
        create_chunk(chunk_id="chunk-1"),
        create_chunk(chunk_id="chunk-2"),
        create_chunk(chunk_id="chunk-3"),
    ]

    result = builder.build(chunks)

    assert len(result) == 3


def test_chunk_id_is_preserved():
    builder = MetadataBuilder()

    chunk = create_chunk(chunk_id="chunk-123")

    result = builder.build([chunk])

    assert result[0].chunk_id == "chunk-123"


def test_eku_id_is_preserved():
    builder = MetadataBuilder()

    chunk = create_chunk(eku_id="eku-456")

    result = builder.build([chunk])

    assert result[0].eku_id == "eku-456"


def test_chunk_text_is_preserved():
    builder = MetadataBuilder()

    text = "Accounting records and classifies financial transactions."

    chunk = create_chunk(text=text)

    result = builder.build([chunk])

    assert result[0].text == text


def test_hierarchy_metadata_is_preserved():
    builder = MetadataBuilder()

    chunk = create_chunk()

    result = builder.build([chunk])

    metadata = result[0].metadata

    assert metadata.level == DocumentLevel.FOUNDATION
    assert metadata.paper == "Paper 1"
    assert metadata.module == "Module 1"
    assert metadata.chapter == "Chapter 1"
    assert metadata.unit == "Unit 1"
    assert metadata.topic == "Introduction"


def test_educational_type_is_preserved():
    builder = MetadataBuilder()

    chunk = create_chunk()

    result = builder.build([chunk])

    assert result[0].metadata.educational_type == EducationalType.CONCEPT


def test_page_provenance_is_preserved():
    builder = MetadataBuilder()

    chunk = create_chunk()

    result = builder.build([chunk])

    assert result[0].metadata.page_start == 2
    assert result[0].metadata.page_end == 3


def test_structural_flags_are_preserved():
    builder = MetadataBuilder()

    chunk = create_chunk()

    result = builder.build([chunk])

    metadata = result[0].metadata

    assert metadata.has_table is False
    assert metadata.has_image is False
    assert metadata.has_formula is False
    assert metadata.has_example is False


def test_metadata_version_is_updated():
    builder = MetadataBuilder()

    chunk = create_chunk()

    result = builder.build([chunk])

    assert result[0].metadata.version == "metadata-v2"


def test_original_metadata_is_not_mutated():
    builder = MetadataBuilder()

    chunk = create_chunk()

    original_version = chunk.metadata.version

    result = builder.build([chunk])

    assert original_version == "initial"
    assert result[0].metadata.version == "metadata-v2"


def test_builder_returns_document_chunks():
    builder = MetadataBuilder()

    chunk = create_chunk()

    result = builder.build([chunk])

    assert isinstance(result[0], DocumentChunk)


def test_builder_does_not_change_chunk_order():
    builder = MetadataBuilder()

    chunks = [
        create_chunk(chunk_id="chunk-1"),
        create_chunk(chunk_id="chunk-2"),
        create_chunk(chunk_id="chunk-3"),
    ]

    result = builder.build(chunks)

    assert [chunk.chunk_id for chunk in result] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]


def test_chunk_indexes_are_preserved():
    builder = MetadataBuilder()

    chunk_1 = create_chunk(chunk_id="chunk-1")
    chunk_1 = chunk_1.model_copy(
        update={
            "chunk_index": 0,
            "total_chunks": 3,
        }
    )

    chunk_2 = create_chunk(chunk_id="chunk-2")
    chunk_2 = chunk_2.model_copy(
        update={
            "chunk_index": 1,
            "total_chunks": 3,
        }
    )

    chunk_3 = create_chunk(chunk_id="chunk-3")
    chunk_3 = chunk_3.model_copy(
        update={
            "chunk_index": 2,
            "total_chunks": 3,
        }
    )

    result = builder.build(
        [
            chunk_1,
            chunk_2,
            chunk_3,
        ]
    )

    assert [chunk.chunk_index for chunk in result] == [0, 1, 2]
    assert [chunk.total_chunks for chunk in result] == [3, 3, 3]


def test_token_count_is_preserved():
    builder = MetadataBuilder()

    chunk = create_chunk()

    result = builder.build([chunk])

    assert result[0].token_count == 10