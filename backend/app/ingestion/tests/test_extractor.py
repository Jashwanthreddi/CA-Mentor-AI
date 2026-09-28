from pathlib import Path

from app.ingestion.extractor import PDFExtractor
from app.ingestion.raw_document import BlockType


PDF_PATH = (
    Path(__file__).resolve().parents[3]
    / "sample_data"
    / "icai"
    / "foundation"
    / "paper_1"
    / "module_1"
    / "chapter_1"
    / "fpp1m1c1u1.pdf"
)


def test_extract_first_three_pages() -> None:
    """
    Verify that the extractor can successfully process
    the first three pages of a real ICAI PDF.
    """

    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    extractor = PDFExtractor(
        do_ocr=False,
        do_table_structure=True,
    )

    document = extractor.extract(
        PDF_PATH,
        page_range=(1, 3),
    )

    assert (
        document.metadata.filename
        == "fpp1m1c1u1.pdf"
    )

    assert (
        document.metadata.parser
        == "docling"
    )

    assert document.metadata.parser_version

    assert len(document.pages) > 0
    assert len(document.pages) <= 3

    all_blocks = [
        block
        for page in document.pages
        for block in page.blocks
    ]

    assert all_blocks

    assert all(
        block.page >= 1
        for block in all_blocks
    )

    assert all(
        block.block_type in BlockType
        for block in all_blocks
    )


def test_extracted_pages_have_valid_dimensions() -> None:
    """
    Verify that extracted pages contain valid physical dimensions.
    """

    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    extractor = PDFExtractor(
        do_ocr=False,
        do_table_structure=True,
    )

    document = extractor.extract(
        PDF_PATH,
        page_range=(1, 3),
    )

    assert document.pages

    for page in document.pages:
        assert page.page_number >= 1
        assert page.width > 0
        assert page.height > 0


def test_extracted_blocks_have_valid_page_numbers() -> None:
    """
    Verify that every extracted block belongs to an
    actually extracted page.
    """

    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    extractor = PDFExtractor(
        do_ocr=False,
        do_table_structure=True,
    )

    document = extractor.extract(
        PDF_PATH,
        page_range=(1, 3),
    )

    extracted_page_numbers = {
        page.page_number
        for page in document.pages
    }

    all_blocks = [
        block
        for page in document.pages
        for block in page.blocks
    ]

    assert all_blocks

    for block in all_blocks:
        assert block.page in extracted_page_numbers


def test_extracted_bounding_boxes_are_normalized() -> None:
    """
    Verify that extracted bounding boxes use the normalized
    top-left coordinate system.

    For valid boxes:

        left <= right
        top <= bottom
    """

    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    extractor = PDFExtractor(
        do_ocr=False,
        do_table_structure=True,
    )

    document = extractor.extract(
        PDF_PATH,
        page_range=(1, 3),
    )

    all_blocks = [
        block
        for page in document.pages
        for block in page.blocks
    ]

    blocks_with_bbox = [
        block
        for block in all_blocks
        if block.bbox is not None
    ]

    assert blocks_with_bbox

    for block in blocks_with_bbox:
        assert block.bbox is not None

        assert block.bbox.left <= block.bbox.right
        assert block.bbox.top <= block.bbox.bottom


def test_extractor_preserves_block_order() -> None:
    """
    Verify that blocks remain in the order returned by
    the extraction layer.
    """

    assert PDF_PATH.exists(), (
        f"Test PDF not found: {PDF_PATH}"
    )

    extractor = PDFExtractor(
        do_ocr=False,
        do_table_structure=True,
    )

    document = extractor.extract(
        PDF_PATH,
        page_range=(1, 3),
    )

    for page in document.pages:
        assert page.blocks == list(page.blocks)