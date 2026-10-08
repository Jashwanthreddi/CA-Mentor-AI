from pathlib import Path

from app.ingestion.extractor import PDFExtractor


PDF_PATH = Path(
    "backend/sample_data/icai/foundation/paper_1/"
    "module_1/chapter_1/fpp1m1c1u1.pdf"
)


def main() -> None:
    extractor = PDFExtractor()

    print("=" * 80)
    print("CA MENTOR AI — PROBLEM PAGE INSPECTION")
    print("=" * 80)

    print(f"PDF: {PDF_PATH}")

    document = extractor.extract(PDF_PATH)

    target_pages = {25, 30}

    for page in document.pages:

        if page.page_number not in target_pages:
            continue

        print("\n" + "=" * 80)
        print(f"PAGE {page.page_number}")
        print("=" * 80)

        print(
            f"Dimensions: "
            f"{page.width:.2f} x {page.height:.2f}"
        )

        print(f"Blocks: {len(page.blocks)}")

        if not page.blocks:
            print("NO RAW BLOCKS FOUND ON THIS PAGE.")
            continue

        for index, block in enumerate(page.blocks, start=1):

            print(
                f"\n[{index}] "
                f"type={block.block_type.value}"
            )

            print(f"text={block.text!r}")

            if block.bbox:
                print(
                    "bbox="
                    f"({block.bbox.left:.1f}, "
                    f"{block.bbox.top:.1f}, "
                    f"{block.bbox.right:.1f}, "
                    f"{block.bbox.bottom:.1f})"
                )

            if block.metadata:
                print(
                    f"metadata={block.metadata}"
                )

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()