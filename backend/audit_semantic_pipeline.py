from pathlib import Path
from collections import Counter

from app.ingestion.extractor import PDFExtractor
from app.ingestion.classifier import EducationalClassifier
from app.ingestion.parser import EducationalParser
from app.ingestion.chunker import SemanticChunker


PDF_PATH = Path(
    "backend/sample_data/icai/foundation/paper_1/"
    "module_1/chapter_1/fpp1m1c1u1.pdf"
)


def main() -> None:
    print("=" * 80)
    print("CA MENTOR AI — FULL PDF SEMANTIC PIPELINE AUDIT")
    print("=" * 80)

    # --------------------------------------------------------------
    # 1. Extraction
    # --------------------------------------------------------------

    print("\n[1/4] Extracting PDF...")

    extractor = PDFExtractor(
        do_ocr=False,
        do_table_structure=False,
        page_batch_size=5,
        retry_failed_pages=True,
        max_page_attempts=2,
    )

    raw_document = extractor.extract(PDF_PATH)

    print(
        f"Extracted pages : {len(raw_document.pages)}"
    )

    total_blocks = sum(
        len(page.blocks)
        for page in raw_document.pages
    )

    print(f"Raw blocks      : {total_blocks}")

    # --------------------------------------------------------------
    # 2. Classification
    # --------------------------------------------------------------

    print("\n[2/4] Classifying educational elements...")

    classifier = EducationalClassifier()

    elements = classifier.classify(raw_document)

    print(f"Elements        : {len(elements)}")

    element_types = Counter(
        element.educational_type.value
        for element in elements
    )

    print("\nEducational element types:")
    for element_type, count in sorted(element_types.items()):
        print(f"  {element_type:25s} {count:5d}")

    # --------------------------------------------------------------
    # 3. Parsing
    # --------------------------------------------------------------

    print("\n[3/4] Parsing knowledge units...")

    parser = EducationalParser()

    knowledge_units = parser.parse(elements)

    print(f"Knowledge units : {len(knowledge_units)}")

    eku_types = Counter(
        eku.educational_type.value
        for eku in knowledge_units
    )

    print("\nKnowledge unit types:")
    for eku_type, count in sorted(eku_types.items()):
        print(f"  {eku_type:25s} {count:5d}")

    # --------------------------------------------------------------
    # 4. Chunking
    # --------------------------------------------------------------

    print("\n[4/4] Creating retrieval chunks...")

    chunker = SemanticChunker()

    chunks = chunker.chunk(knowledge_units)

    print(f"Chunks          : {len(chunks)}")

    chunk_types = Counter(
        chunk.metadata.educational_type.value
        for chunk in chunks
    )

    print("\nChunk educational types:")
    for chunk_type, count in sorted(chunk_types.items()):
        print(f"  {chunk_type:25s} {count:5d}")

    # --------------------------------------------------------------
    # Page coverage
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PAGE COVERAGE")
    print("=" * 80)

    page_element_counts = Counter()

    for element in elements:
        for page in range(
            element.page_start,
            element.page_end + 1,
        ):
            page_element_counts[page] += 1

    for page_number in range(
        1,
        len(raw_document.pages) + 1,
    ):
        print(
            f"Page {page_number:02d} | "
            f"elements={page_element_counts[page_number]:3d}"
        )

    # --------------------------------------------------------------
    # Hierarchy
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("HIERARCHY SAMPLE")
    print("=" * 80)

    for index, element in enumerate(elements[:30], start=1):
        hierarchy = " > ".join(
            element.hierarchy_path
        )

        preview = " ".join(
            element.content.split()
        )[:100]

        print(
            f"{index:02d}. "
            f"[{element.educational_type.value}] "
            f"{hierarchy}"
        )
        print(f"    {preview}")

    # --------------------------------------------------------------
    # Knowledge unit sample
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("KNOWLEDGE UNIT SAMPLE")
    print("=" * 80)

    for index, eku in enumerate(
        knowledge_units[:20],
        start=1,
    ):
        hierarchy = " > ".join(
            eku.hierarchy_path
        )

        preview = " ".join(
            eku.content.split()
        )[:150]

        print(
            f"{index:02d}. "
            f"[{eku.educational_type.value}] "
            f"pages={eku.metadata.page_start}-"
            f"{eku.metadata.page_end}"
        )
        print(f"    hierarchy: {hierarchy}")
        print(f"    content  : {preview}")

    # --------------------------------------------------------------
    # Chunk sample
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CHUNK SAMPLE")
    print("=" * 80)

    for index, chunk in enumerate(
        chunks[:20],
        start=1,
    ):
        preview = " ".join(
            chunk.text.split()
        )[:150]

        print(
            f"{index:02d}. "
            f"id={chunk.chunk_id}"
        )
        print(
            f"    type={chunk.metadata.educational_type.value} "
            f"pages={chunk.metadata.page_start}-"
            f"{chunk.metadata.page_end} "
            f"tokens={chunk.token_count}"
        )
        print(f"    {preview}")

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"PDF pages       : {len(raw_document.pages)}")
    print(f"Raw blocks      : {total_blocks}")
    print(f"Elements        : {len(elements)}")
    print(f"Knowledge units : {len(knowledge_units)}")
    print(f"Chunks          : {len(chunks)}")

    print("=" * 80)


if __name__ == "__main__":
    main()