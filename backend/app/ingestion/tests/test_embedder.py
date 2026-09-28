import pytest

from app.ingestion.embedder import DocumentEmbedder
from app.ingestion.models import (
    ChunkMetadata,
    DocumentChunk,
    DocumentLevel,
    EducationalType,
)


class FakeEmbeddingModel:
    """
    Lightweight fake embedding model used for unit tests.

    It avoids downloading or loading the real BGE model.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.calls = []

    def encode(
        self,
        texts,
        batch_size,
        show_progress_bar,
        normalize_embeddings,
    ):
        self.calls.append(
            {
                "texts": texts,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "normalize_embeddings": normalize_embeddings,
            }
        )

        return [
            [0.01] * self.dimension
            for _ in texts
        ]


def create_chunk(
    *,
    chunk_id: str = "chunk-1",
    eku_id: str = "eku-1",
    text: str = "Accounting records business transactions.",
    chunk_index: int = 0,
    total_chunks: int = 1,
    token_count: int = 10,
) -> DocumentChunk:
    metadata = ChunkMetadata(
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

    return DocumentChunk(
        chunk_id=chunk_id,
        eku_id=eku_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        text=text,
        token_count=token_count,
        metadata=metadata,
    )


def test_empty_input_returns_empty_list():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    result = embedder.embed([])

    assert result == []
    assert model.calls == []


def test_single_chunk_is_embedded():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    chunk = create_chunk()

    result = embedder.embed([chunk])

    assert len(result) == 1
    assert len(result[0].embedding) == 768


def test_embedding_dimension_is_768():
    model = FakeEmbeddingModel(dimension=768)

    embedder = DocumentEmbedder(
        model=model,
    )

    result = embedder.embed(
        [create_chunk()]
    )

    assert len(result[0].embedding) == 768


def test_wrong_embedding_dimension_is_rejected():
    model = FakeEmbeddingModel(dimension=384)

    embedder = DocumentEmbedder(
        model=model,
    )

    with pytest.raises(
        ValueError,
        match="Unexpected embedding dimension",
    ):
        embedder.embed(
            [create_chunk()]
        )


def test_chunk_id_is_preserved():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    chunk = create_chunk(
        chunk_id="chunk-123",
    )

    result = embedder.embed([chunk])

    assert result[0].chunk_id == "chunk-123"


def test_eku_id_is_preserved():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    chunk = create_chunk(
        eku_id="eku-123",
    )

    result = embedder.embed([chunk])

    assert result[0].eku_id == "eku-123"


def test_text_is_preserved():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    text = "Accounting information is useful to decision makers."

    chunk = create_chunk(
        text=text,
    )

    result = embedder.embed([chunk])

    assert result[0].text == text


def test_chunk_index_is_preserved():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    chunk = create_chunk(
        chunk_index=2,
        total_chunks=5,
    )

    result = embedder.embed([chunk])

    assert result[0].chunk_index == 2
    assert result[0].total_chunks == 5


def test_token_count_is_preserved():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    chunk = create_chunk(
        token_count=42,
    )

    result = embedder.embed([chunk])

    assert result[0].token_count == 42


def test_metadata_is_preserved():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    chunk = create_chunk()

    result = embedder.embed([chunk])

    assert result[0].metadata == chunk.metadata


def test_multiple_chunks_are_embedded():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    chunks = [
        create_chunk(chunk_id="chunk-1"),
        create_chunk(chunk_id="chunk-2"),
        create_chunk(chunk_id="chunk-3"),
    ]

    result = embedder.embed(chunks)

    assert len(result) == 3
    assert [
        chunk.chunk_id
        for chunk in result
    ] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]


def test_model_receives_chunk_text():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    chunks = [
        create_chunk(
            text="First accounting concept."
        ),
        create_chunk(
            chunk_id="chunk-2",
            text="Second accounting concept."
        ),
    ]

    embedder.embed(chunks)

    assert model.calls[0]["texts"] == [
        "First accounting concept.",
        "Second accounting concept.",
    ]


def test_batch_size_is_passed_to_model():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
        batch_size=8,
    )

    embedder.embed(
        [create_chunk()]
    )

    assert model.calls[0]["batch_size"] == 8


def test_embeddings_are_requested_normalized():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    embedder.embed(
        [create_chunk()]
    )

    assert model.calls[0]["normalize_embeddings"] is True


def test_progress_bar_is_disabled():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    embedder.embed(
        [create_chunk()]
    )

    assert model.calls[0]["show_progress_bar"] is False


def test_embedding_values_are_float():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    result = embedder.embed(
        [create_chunk()]
    )

    assert all(
        isinstance(value, float)
        for value in result[0].embedding
    )


def test_model_is_injected_without_loading_real_model():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    assert embedder._model is model


def test_default_model_name_is_bge_base():
    embedder = DocumentEmbedder(
        model=FakeEmbeddingModel(),
    )

    assert (
        embedder.model_name
        == "BAAI/bge-base-en-v1.5"
    )


def test_default_batch_size_is_16():
    embedder = DocumentEmbedder(
        model=FakeEmbeddingModel(),
    )

    assert embedder.batch_size == 16


def test_empty_model_name_is_rejected():
    with pytest.raises(
        ValueError,
        match="model_name cannot be empty",
    ):
        DocumentEmbedder(
            model_name="",
            model=FakeEmbeddingModel(),
        )


def test_invalid_batch_size_is_rejected():
    with pytest.raises(
        ValueError,
        match="batch_size must be at least 1",
    ):
        DocumentEmbedder(
            batch_size=0,
            model=FakeEmbeddingModel(),
        )


def test_input_order_is_preserved():
    model = FakeEmbeddingModel()

    embedder = DocumentEmbedder(
        model=model,
    )

    chunks = [
        create_chunk(chunk_id="chunk-A"),
        create_chunk(chunk_id="chunk-B"),
        create_chunk(chunk_id="chunk-C"),
    ]

    result = embedder.embed(chunks)

    assert [
        chunk.chunk_id
        for chunk in result
    ] == [
        "chunk-A",
        "chunk-B",
        "chunk-C",
    ]