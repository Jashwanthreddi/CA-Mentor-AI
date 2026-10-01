from pathlib import Path

from app.ingestion.classifier import EducationalClassifier
from app.ingestion.extractor import PDFExtractor


PDF_PATH = (
    Path(__file__).resolve().parent
    / "sample_data"
    / "icai"
    / "foundation"
    / "paper_1"
    / "module_1"
    / "chapter_1"
    / "fpp1m1c1u1.pdf"
)


def main() -> None:
    extractor = PDFExtractor(
        do_ocr=False,
        do_table_structure=True,
    )

    print("\n" + "=" * 80)
    print("EXTRACTING ICAI PDF")
    print("=" * 80)

    document = extractor.extract(
        PDF_PATH,
        page_range=(1, 3),
    )

    print("\n")
    print("=" * 80)
    print("RAW EXTRACTED DATA")
    print("=" * 80)

    for page in document.pages:
        print(f"\nPAGE {page.page_number}")
        print("-" * 80)

        for index, block in enumerate(page.blocks, start=1):
            print(f"\n[{index}] {block.block_type.value}")
            print(f"ID: {block.id}")
            print(f"TEXT: {block.text}")

            if block.bbox:
                print(
                    "BBOX: "
                    f"left={block.bbox.left:.2f}, "
                    f"top={block.bbox.top:.2f}, "
                    f"right={block.bbox.right:.2f}, "
                    f"bottom={block.bbox.bottom:.2f}"
                )

    classifier = EducationalClassifier()

    elements = classifier.classify(document)

    print("\n")
    print("=" * 80)
    print("CLASSIFIED EDUCATIONAL ELEMENTS")
    print("=" * 80)

    for index, element in enumerate(elements, start=1):
        print(f"\n[{index}]")
        print(f"TYPE: {element.educational_type.value}")
        print(f"CONFIDENCE: {element.confidence}")
        print(f"PAGE: {element.page_start}-{element.page_end}")
        print(f"HIERARCHY: {element.hierarchy_path}")
        print(f"SOURCE BLOCKS: {element.source_block_ids}")
        print(f"CONTENT: {element.content}")


if __name__ == "__main__":
    main()