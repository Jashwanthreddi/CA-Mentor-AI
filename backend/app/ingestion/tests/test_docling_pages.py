from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)


pdf_path = Path(
    "backend/sample_data/icai/foundation/paper_1/"
    "module_1/chapter_1/fpp1m1c1u1.pdf"
)


pipeline_options = PdfPipelineOptions()

pipeline_options.do_ocr = False
pipeline_options.do_table_structure = False


converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)


print("Testing pages 1-5...")

result = converter.convert(
    pdf_path,
    page_range=(1, 5),
)

document = result.document

print("Pages returned:", len(document.pages))

for page_number, page in document.pages.items():
    print(
        f"page={page_number}"
    )


print("\nTesting pages 6-10...")

result = converter.convert(
    pdf_path,
    page_range=(6, 10),
)

document = result.document

print("Pages returned:", len(document.pages))

for page_number, page in document.pages.items():
    print(
        f"page={page_number}"
    )


print("\nTesting pages 8-8...")

result = converter.convert(
    pdf_path,
    page_range=(8, 8),
)

document = result.document

print("Pages returned:", len(document.pages))

for page_number, page in document.pages.items():
    print(
        f"page={page_number}"
    )