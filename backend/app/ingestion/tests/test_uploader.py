"""
Tests for the Qdrant uploader.

These tests verify:

    - collection creation
    - collection reuse
    - Qdrant-compatible point IDs
    - payload construction
    - embedding validation
    - empty uploads
    - upsert behavior
    - idempotent behavior
"""

from unittest.mock import MagicMock

import pytest
from qdrant_client.models import PointStruct

from app.ingestion.models import (
    ChunkMetadata,
    DocumentLevel,
    EducationalType,
    EmbeddedChunk,
)
from app.ingestion.uploader import QdrantUploader


def create_embedded_chunk(
    *,
    chunk_id: str = (
        "chunk-12345678-1234-1234-1234-123456789abc"
    ),
    eku_id: str = "eku-1",
    chunk_index: int = 0,
    total_chunks: int = 1,
    text: str = (
        "Accounting records business transactions."
    ),
    token_count: int = 10,
) -> EmbeddedChunk:
    """Create a valid EmbeddedChunk for uploader tests."""

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
        page_start=1,
        page_end=1,
        has_table=False,
        has_image=False,
        has_formula=False,
        has_example=False,
        version="metadata-v1",
    )

    return EmbeddedChunk(
        chunk_id=chunk_id,
        eku_id=eku_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        text=text,
        token_count=token_count,
        metadata=metadata,
        embedding=[0.1] * 768,
    )


def create_uploader(
    client: MagicMock | None = None,
) -> QdrantUploader:
    """Create a QdrantUploader with an injected client."""

    if client is None:
        client = MagicMock()

    return QdrantUploader(
        client=client,
        collection_name="test_collection",
    )


def test_empty_collection_name_is_rejected():
    with pytest.raises(
        ValueError,
        match="collection_name cannot be empty",
    ):
        QdrantUploader(
            client=MagicMock(),
            collection_name="",
        )


def test_invalid_vector_dimension_is_rejected():
    with pytest.raises(
        ValueError,
        match="vector_dimension must be at least 1",
    ):
        QdrantUploader(
            client=MagicMock(),
            collection_name="test_collection",
            vector_dimension=0,
        )


def test_collection_is_created_when_missing():
    client = MagicMock()

    client.collection_exists.return_value = False

    uploader = create_uploader(client)

    uploader.ensure_collection()

    client.collection_exists.assert_called_once()
    client.create_collection.assert_called_once()


def test_existing_collection_is_not_created_again():
    client = MagicMock()

    # The uploader checks the result of collection_exists().
    client.collection_exists.return_value = True

    uploader = create_uploader(client)

    uploader.ensure_collection()

    client.collection_exists.assert_called_once()
    client.create_collection.assert_not_called()


def test_payload_preserves_chunk_identity():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    payload = uploader._build_payload(chunk)

    assert payload["chunk_id"] == chunk.chunk_id
    assert payload["eku_id"] == chunk.eku_id


def test_payload_preserves_text():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    payload = uploader._build_payload(chunk)

    assert payload["text"] == chunk.text


def test_payload_preserves_academic_hierarchy():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    payload = uploader._build_payload(chunk)

    assert payload["level"] == "foundation"
    assert payload["paper"] == "Paper 1"
    assert payload["module"] == "Module 1"
    assert payload["chapter"] == "Chapter 1"
    assert payload["unit"] == "Unit 1"
    assert payload["topic"] == "Introduction"


def test_payload_preserves_educational_metadata():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    payload = uploader._build_payload(chunk)

    assert (
        payload["educational_type"]
        == EducationalType.CONCEPT.value
    )


def test_payload_preserves_provenance():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    payload = uploader._build_payload(chunk)

    assert payload["page_start"] == 1
    assert payload["page_end"] == 1


def test_payload_preserves_content_flags():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    payload = uploader._build_payload(chunk)

    assert payload["has_table"] is False
    assert payload["has_image"] is False
    assert payload["has_formula"] is False
    assert payload["has_example"] is False


def test_point_uses_uuid_as_qdrant_id():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    point = uploader._build_point(chunk)

    assert isinstance(point, PointStruct)

    assert (
        point.id
        == "12345678-1234-1234-1234-123456789abc"
    )


def test_point_preserves_original_chunk_id_in_payload():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    point = uploader._build_point(chunk)

    assert point.payload["chunk_id"] == chunk.chunk_id


def test_point_contains_embedding():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    point = uploader._build_point(chunk)

    assert point.vector == chunk.embedding


def test_point_contains_payload():
    chunk = create_embedded_chunk()

    uploader = create_uploader()

    point = uploader._build_point(chunk)

    assert point.payload is not None
    assert point.payload["chunk_id"] == chunk.chunk_id
    assert point.payload["eku_id"] == chunk.eku_id


def test_invalid_chunk_id_is_rejected():
    chunk = create_embedded_chunk(
        chunk_id="chunk-invalid"
    )

    uploader = create_uploader()

    with pytest.raises(
        ValueError,
        match="valid UUID",
    ):
        uploader._build_point(chunk)


def test_chunk_id_without_prefix_is_rejected():
    chunk = create_embedded_chunk(
        chunk_id=(
            "12345678-1234-1234-1234-123456789abc"
        )
    )

    uploader = create_uploader()

    with pytest.raises(
        ValueError,
        match="Invalid chunk ID format",
    ):
        uploader._build_point(chunk)


def test_wrong_embedding_dimension_is_rejected():
    chunk = create_embedded_chunk()

    chunk.embedding = [0.1] * 100

    uploader = create_uploader()

    with pytest.raises(
        ValueError,
        match="768",
    ):
        uploader._build_point(chunk)


def test_non_numeric_embedding_is_rejected():
    chunk = create_embedded_chunk()

    chunk.embedding = ["invalid"] * 768

    uploader = create_uploader()

    with pytest.raises(
        TypeError,
        match="numeric",
    ):
        uploader._build_point(chunk)


def test_empty_upload_returns_successful_report():
    client = MagicMock()

    uploader = create_uploader(client)

    report = uploader.upload(
        [],
        filename="test.pdf",
        total_pages=1,
        sections=1,
        eku_count=1,
        tables=0,
        images=0,
        embedding_time=0.1,
    )

    assert report.success is True
    assert report.chunks == 0

    client.upsert.assert_not_called()


def test_upload_creates_collection_and_upserts_chunks():
    client = MagicMock()

    client.collection_exists.return_value = False

    uploader = create_uploader(client)

    chunks = [
        create_embedded_chunk(),
    ]

    report = uploader.upload(
        chunks,
        filename="test.pdf",
        total_pages=1,
        sections=1,
        eku_count=1,
        tables=0,
        images=0,
        embedding_time=0.1,
    )

    client.create_collection.assert_called_once()
    client.upsert.assert_called_once()

    assert report.success is True
    assert report.chunks == 1


def test_upload_preserves_chunk_order():
    client = MagicMock()

    client.collection_exists.return_value = True

    uploader = create_uploader(client)

    chunk_1 = create_embedded_chunk(
        chunk_id=(
            "chunk-11111111-1111-1111-1111-111111111111"
        ),
        chunk_index=0,
    )

    chunk_2 = create_embedded_chunk(
        chunk_id=(
            "chunk-22222222-2222-2222-2222-222222222222"
        ),
        chunk_index=1,
        total_chunks=2,
    )

    uploader.upload(
        [chunk_1, chunk_2],
        filename="test.pdf",
        total_pages=1,
        sections=1,
        eku_count=1,
        tables=0,
        images=0,
        embedding_time=0.1,
    )

    call = client.upsert.call_args

    points = call.kwargs["points"]

    assert len(points) == 2

    assert (
        points[0].id
        == "11111111-1111-1111-1111-111111111111"
    )

    assert (
        points[1].id
        == "22222222-2222-2222-2222-222222222222"
    )


def test_upload_is_idempotent_for_same_chunk_ids():
    client = MagicMock()

    client.collection_exists.return_value = True

    uploader = create_uploader(client)

    chunk = create_embedded_chunk()

    uploader.upload(
        [chunk],
        filename="test.pdf",
        total_pages=1,
        sections=1,
        eku_count=1,
        tables=0,
        images=0,
        embedding_time=0.1,
    )

    uploader.upload(
        [chunk],
        filename="test.pdf",
        total_pages=1,
        sections=1,
        eku_count=1,
        tables=0,
        images=0,
        embedding_time=0.1,
    )

    assert client.upsert.call_count == 2