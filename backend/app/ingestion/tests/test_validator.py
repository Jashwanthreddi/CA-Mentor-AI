from app.ingestion.models import (
    ChunkMetadata,
    DocumentChunk,
    DocumentLevel,
    EducationalType,
)
import pytest
from app.ingestion.validator import (
    DocumentChunkValidator,
    ValidationResult,
)


def create_chunk(
    *,
    chunk_id: str = "chunk-1",
    eku_id: str = "eku-1",
    text: str = "Accounting records business transactions.",
    chunk_index: int = 0,
    total_chunks: int = 1,
    token_count: int = 10,
    page_start: int = 1,
    page_end: int = 1,
) -> DocumentChunk:
    """
    Create a valid DocumentChunk for validator tests.
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
        page_start=page_start,
        page_end=page_end,
        has_table=False,
        has_image=False,
        has_formula=False,
        has_example=False,
        version="metadata-v1",
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        eku_id=eku_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        text=text,
        token_count=token_count,
        metadata=metadata,
    )


def test_empty_input_is_valid():
    validator = DocumentChunkValidator()

    result = validator.validate([])

    assert isinstance(result, ValidationResult)
    assert result.total_chunks == 0
    assert result.valid_count == 0
    assert result.invalid_count == 0
    assert result.is_valid is True


def test_valid_chunk_is_accepted():
    validator = DocumentChunkValidator()

    chunk = create_chunk()

    result = validator.validate([chunk])

    assert result.valid_count == 1
    assert result.invalid_count == 0
    assert result.valid_chunks == [chunk]


def test_valid_chunk_is_not_mutated():
    validator = DocumentChunkValidator()

    chunk = create_chunk()

    result = validator.validate([chunk])

    assert result.valid_chunks[0] == chunk


def test_empty_text_is_rejected():
    validator = DocumentChunkValidator()

    chunk = create_chunk(text="")

    result = validator.validate([chunk])

    assert result.invalid_count == 1
    assert result.valid_count == 0
    assert any(
        "chunk text cannot be empty" in error
        for error in result.errors
    )


def test_whitespace_text_is_rejected():
    validator = DocumentChunkValidator()

    chunk = create_chunk(text="   ")

    result = validator.validate([chunk])

    assert result.invalid_count == 1
    assert any(
        "chunk text cannot be empty" in error
        for error in result.errors
    )


def test_empty_chunk_id_is_rejected():
    validator = DocumentChunkValidator()

    chunk = create_chunk(chunk_id="")

    result = validator.validate([chunk])

    assert result.invalid_count == 1
    assert any(
        "chunk_id cannot be empty" in error
        for error in result.errors
    )


def test_empty_eku_id_is_rejected():
    validator = DocumentChunkValidator()

    chunk = create_chunk(eku_id="")

    result = validator.validate([chunk])

    assert result.invalid_count == 1
    assert any(
        "eku_id cannot be empty" in error
        for error in result.errors
    )


def test_duplicate_chunk_ids_are_rejected():
    validator = DocumentChunkValidator()

    chunk_1 = create_chunk(chunk_id="chunk-1")
    chunk_2 = create_chunk(chunk_id="chunk-1")

    result = validator.validate([chunk_1, chunk_2])

    assert result.valid_count == 1
    assert result.invalid_count == 1

    assert any(
        "duplicate chunk_id detected" in error
        for error in result.errors
    )


def test_chunk_index_zero_is_valid():
    validator = DocumentChunkValidator()

    chunk = create_chunk(
        chunk_index=0,
        total_chunks=3,
    )

    result = validator.validate([chunk])

    assert result.valid_count == 1


def test_chunk_index_equal_to_total_chunks_is_rejected():
    validator = DocumentChunkValidator()

    chunk = create_chunk(
        chunk_index=3,
        total_chunks=3,
    )

    result = validator.validate([chunk])

    assert result.invalid_count == 1
    assert any(
        "chunk_index must be smaller than total_chunks"
        in error
        for error in result.errors
    )


def test_negative_token_count_is_rejected_by_model():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        create_chunk(token_count=-1)

def test_zero_token_count_is_allowed_structurally():
    validator = DocumentChunkValidator()

    chunk = create_chunk(token_count=0)

    result = validator.validate([chunk])

    assert result.valid_count == 1


def test_page_range_is_valid():
    validator = DocumentChunkValidator()

    chunk = create_chunk(
        page_start=3,
        page_end=5,
    )

    result = validator.validate([chunk])

    assert result.valid_count == 1


def test_invalid_page_range_is_rejected():
    validator = DocumentChunkValidator()

    chunk = create_chunk(
        page_start=5,
        page_end=3,
    )

    result = validator.validate([chunk])

    assert result.invalid_count == 1
    assert any(
        "page_start cannot be greater than page_end"
        in error
        for error in result.errors
    )


def test_empty_paper_is_rejected():
    validator = DocumentChunkValidator()

    chunk = create_chunk()

    chunk = chunk.model_copy(
        update={
            "metadata": chunk.metadata.model_copy(
                update={
                    "paper": "",
                }
            )
        }
    )

    result = validator.validate([chunk])

    assert result.invalid_count == 1
    assert any(
        "paper cannot be empty" in error
        for error in result.errors
    )


def test_empty_metadata_version_is_rejected():
    validator = DocumentChunkValidator()

    chunk = create_chunk()

    chunk = chunk.model_copy(
        update={
            "metadata": chunk.metadata.model_copy(
                update={
                    "version": "",
                }
            )
        }
    )

    result = validator.validate([chunk])

    assert result.invalid_count == 1
    assert any(
        "metadata version cannot be empty" in error
        for error in result.errors
    )


def test_valid_and_invalid_chunks_are_separated():
    validator = DocumentChunkValidator()

    valid_chunk = create_chunk(
        chunk_id="valid-chunk",
    )

    invalid_chunk = create_chunk(
        chunk_id="invalid-chunk",
        text="",
    )

    result = validator.validate(
        [
            valid_chunk,
            invalid_chunk,
        ]
    )

    assert result.valid_count == 1
    assert result.invalid_count == 1
    assert result.valid_chunks == [valid_chunk]
    assert result.invalid_chunks == [invalid_chunk]


def test_multiple_invalid_chunks_are_collected():
    validator = DocumentChunkValidator()

    chunk_1 = create_chunk(
        chunk_id="chunk-1",
        text="",
    )

    chunk_2 = create_chunk(
        chunk_id="chunk-2",
        eku_id="",
    )

    result = validator.validate(
        [
            chunk_1,
            chunk_2,
        ]
    )

    assert result.valid_count == 0
    assert result.invalid_count == 2
    assert len(result.errors) >= 2


def test_total_count_is_consistent():
    validator = DocumentChunkValidator()

    chunks = [
        create_chunk(chunk_id="chunk-1"),
        create_chunk(chunk_id="chunk-2"),
        create_chunk(
            chunk_id="chunk-3",
            text="",
        ),
    ]

    result = validator.validate(chunks)

    assert result.total_chunks == 3
    assert (
        result.valid_count + result.invalid_count
        == result.total_chunks
    )


def test_is_valid_is_false_when_invalid_chunks_exist():
    validator = DocumentChunkValidator()

    chunk = create_chunk(text="")

    result = validator.validate([chunk])

    assert result.is_valid is False


def test_is_valid_is_true_when_all_chunks_are_valid():
    validator = DocumentChunkValidator()

    chunks = [
        create_chunk(chunk_id="chunk-1"),
        create_chunk(chunk_id="chunk-2"),
    ]

    result = validator.validate(chunks)

    assert result.is_valid is True


def test_chunk_order_is_preserved():
    validator = DocumentChunkValidator()

    chunks = [
        create_chunk(chunk_id="chunk-1"),
        create_chunk(chunk_id="chunk-2"),
        create_chunk(chunk_id="chunk-3"),
    ]

    result = validator.validate(chunks)

    assert [
        chunk.chunk_id
        for chunk in result.valid_chunks
    ] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]