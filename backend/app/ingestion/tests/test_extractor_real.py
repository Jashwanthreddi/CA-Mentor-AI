from pathlib import Path

from app.ingestion.extractor import PDFExtractor


pdf_path = Path(
    "backend/sample_data/icai/foundation/paper_1/"
    "module_1/chapter_1/fpp1m1c1u1.pdf"
)


extractor = PDFExtractor(
    do_ocr=False,
    do_table_structure=False,
    page_batch_size=5,
    retry_failed_pages=True,
    max_page_attempts=2,
)


document = extractor.extract(pdf_path)

print("\n" + "=" * 70)
print("EXTRACTION RESULT")
print("=" * 70)

print(f"Filename: {document.metadata.filename}")
print(f"Pages: {document.metadata.pages}")
print(f"Parser: {document.metadata.parser}")
print(
    f"Parser version: "
    f"{document.metadata.parser_version}"
)

print("\nPAGE STATISTICS")

for page in document.pages:
    characters = sum(
        len(block.text)
        for block in page.blocks
    )

    print(
        f"Page {page.page_number:02d} | "
        f"blocks={len(page.blocks):3d} | "
        f"characters={characters:5d}"
    )

print("=" * 70)