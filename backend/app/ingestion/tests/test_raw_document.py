from app.ingestion.raw_document import (
    BlockType,
    BoundingBox,
    RawBlock,
    RawDocument,
    RawDocumentMetadata,
    RawPage,
)


def test_raw_document_creation() -> None:
    block = RawBlock(
        id="block-001",
        page=1,
        block_type=BlockType.HEADING,
        text="Meaning of Accounting",
        bbox=BoundingBox(
            left=50,
            top=100,
            right=500,
            bottom=140,
        ),
    )

    page = RawPage(
        page_number=1,
        width=612,
        height=792,
        blocks=[block],
    )

    metadata = RawDocumentMetadata(
        filename="accounting_unit1.pdf",
        title="Meaning and Scope of Accounting",
        pages=1,
        language="English",
        parser="docling",
        parser_version="2.115.0",
    )

    document = RawDocument(
        metadata=metadata,
        pages=[page],
    )

    assert document.metadata.filename == "accounting_unit1.pdf"
    assert len(document.pages) == 1
    assert len(document.pages[0].blocks) == 1
    assert document.pages[0].blocks[0].block_type == BlockType.HEADING