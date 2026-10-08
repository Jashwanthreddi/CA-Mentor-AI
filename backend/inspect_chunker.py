from pathlib import Path
from collections import Counter
from statistics import mean

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
    print("CA MENTOR AI - REAL ICAI CHUNKER AUDIT")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. EXTRACTION
    # ---------------------------------------------------------

    print("\n[1] Extracting PDF...")

    extractor = PDFExtractor(
        do_ocr=False,
        do_table_structure=True,
    )

    raw_document = extractor.extract(PDF_PATH)

    print(f"Pages : {len(raw_document.pages)}")
    print(
        f"Blocks: "
        f"{sum(len(page.blocks) for page in raw_document.pages)}"
    )

    # ---------------------------------------------------------
    # 2. CLASSIFICATION
    # ---------------------------------------------------------

    print("\n[2] Classifying educational elements...")

    classifier = EducationalClassifier()
    elements = classifier.classify(raw_document)

    print(f"Educational elements: {len(elements)}")

    element_types = Counter(
        element.educational_type.value
        for element in elements
    )

    print("\nElement types:")
    for name, count in element_types.most_common():
        print(f"  {name:25} {count}")

    # ---------------------------------------------------------
    # 3. PARSING
    # ---------------------------------------------------------

    print("\n[3] Parsing into EKUs...")

    parser = EducationalParser()
    ekus = parser.parse(elements)

    print(f"EKUs: {len(ekus)}")

    eku_types = Counter(
        eku.educational_type.value
        for eku in ekus
    )

    print("\nEKU types:")
    for name, count in eku_types.most_common():
        print(f"  {name:25} {count}")

    # ---------------------------------------------------------
    # 4. CHUNKING
    # ---------------------------------------------------------

    print("\n[4] Chunking...")

    chunker = SemanticChunker(
        target_tokens=700,
        max_tokens=900,
        overlap_tokens=100,
        min_tokens=100,
    )

    chunks = chunker.chunk(ekus)

    print(f"Chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # 5. SIZE AUDIT
    # ---------------------------------------------------------

    print("\n[5] Chunk size audit...")

    token_counts = [
        chunk.token_count
        for chunk in chunks
    ]

    if token_counts:
        print(f"Minimum tokens : {min(token_counts)}")
        print(f"Maximum tokens : {max(token_counts)}")
        print(f"Average tokens : {mean(token_counts):.2f}")
        print(
            f"Median tokens  : "
            f"{sorted(token_counts)[len(token_counts) // 2]}"
        )

    oversized = [
        chunk
        for chunk in chunks
        if chunk.token_count > chunker.max_tokens
    ]

    tiny = [
        chunk
        for chunk in chunks
        if chunk.token_count < chunker.min_tokens
    ]

    print(f"\nOversized chunks (>900): {len(oversized)}")
    print(f"Tiny chunks (<100)    : {len(tiny)}")

    # ---------------------------------------------------------
    # 6. CHUNK TYPE AUDIT
    # ---------------------------------------------------------

    print("\n[6] Chunk educational types...")

    chunk_types = Counter(
        chunk.metadata.educational_type.value
        for chunk in chunks
    )

    for name, count in chunk_types.most_common():
        print(f"  {name:25} {count}")

    # ---------------------------------------------------------
    # 7. ATOMIC CONTENT AUDIT
    # ---------------------------------------------------------

    print("\n[7] Atomic-content audit...")

    atomic_types = {
        "learning_outcome",
        "practice_question",
        "exam_question",
        "formula",
        "memory_trick",
    }

    atomic_ekus = [
        eku
        for eku in ekus
        if eku.educational_type.value in atomic_types
    ]

    atomic_eku_ids = {
        eku.id
        for eku in atomic_ekus
    }

    atomic_chunks = [
        chunk
        for chunk in chunks
        if chunk.eku_id in atomic_eku_ids
    ]

    print(f"Atomic EKUs    : {len(atomic_ekus)}")
    print(f"Atomic chunks  : {len(atomic_chunks)}")

    for eku in atomic_ekus:
        eku_chunks = [
            chunk
            for chunk in atomic_chunks
            if chunk.eku_id == eku.id
        ]

        print(
            f"  {eku.educational_type.value:20} "
            f"chunks={len(eku_chunks):2} "
            f"title={eku.title[:60]}"
        )

    # ---------------------------------------------------------
    # 8. PROVENANCE AUDIT
    # ---------------------------------------------------------

    print("\n[8] Provenance audit...")

    invalid_provenance = []

    for chunk in chunks:
        if chunk.metadata.page_start < 1:
            invalid_provenance.append(chunk)

        if chunk.metadata.page_end < chunk.metadata.page_start:
            invalid_provenance.append(chunk)

        if not chunk.eku_id:
            invalid_provenance.append(chunk)

    print(
        f"Invalid provenance chunks: "
        f"{len(invalid_provenance)}"
    )

    # ---------------------------------------------------------
    # 9. EMPTY CONTENT AUDIT
    # ---------------------------------------------------------

    print("\n[9] Empty-content audit...")

    empty_chunks = [
        chunk
        for chunk in chunks
        if not chunk.text.strip()
    ]

    print(f"Empty chunks: {len(empty_chunks)}")

    # ---------------------------------------------------------
    # 10. DUPLICATE ID AUDIT
    # ---------------------------------------------------------

    print("\n[10] Chunk ID audit...")

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    duplicate_ids = [
        chunk_id
        for chunk_id, count in Counter(chunk_ids).items()
        if count > 1
    ]

    print(f"Total IDs      : {len(chunk_ids)}")
    print(f"Unique IDs     : {len(set(chunk_ids))}")
    print(f"Duplicate IDs  : {len(duplicate_ids)}")

    # ---------------------------------------------------------
    # 11. SPLIT EKU AUDIT
    # ---------------------------------------------------------

    print("\n[11] EKU split audit...")

    eku_chunk_counts = Counter(
        chunk.eku_id
        for chunk in chunks
    )

    split_ekus = {
        eku_id: count
        for eku_id, count in eku_chunk_counts.items()
        if count > 1
    }

    print(f"EKUs split into multiple chunks: {len(split_ekus)}")

    if split_ekus:
        print("\nTop split EKUs:")

        for eku_id, count in sorted(
            split_ekus.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]:
            matching = next(
                (
                    eku
                    for eku in ekus
                    if eku.id == eku_id
                ),
                None,
            )

            title = matching.title if matching else "Unknown"

            print(
                f"  chunks={count:2} "
                f"title={title[:70]}"
            )

    # ---------------------------------------------------------
    # 12. SAMPLE CHUNKS
    # ---------------------------------------------------------

    print("\n[12] Sample chunks")
    print("-" * 80)

    for index, chunk in enumerate(chunks[:10], start=1):
        print(
            f"\nCHUNK {index}"
        )
        print(
            f"ID       : {chunk.chunk_id}"
        )
        print(
            f"EKU      : {chunk.eku_id}"
        )
        print(
            f"Tokens   : {chunk.token_count}"
        )
        print(
            f"Pages    : "
            f"{chunk.metadata.page_start}-"
            f"{chunk.metadata.page_end}"
        )
        print(
            f"Type     : "
            f"{chunk.metadata.educational_type.value}"
        )
        print(
            f"Hierarchy: "
            f"{chunk.metadata.topic or chunk.metadata.chapter}"
        )
        print(
            f"Text     : "
            f"{chunk.text[:300].replace(chr(10), ' ')}..."
        )

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL AUDIT SUMMARY")
    print("=" * 80)

    checks = {
        "Chunks generated": len(chunks) > 0,
        "No empty chunks": len(empty_chunks) == 0,
        "No duplicate IDs": len(duplicate_ids) == 0,
        "Valid provenance": len(invalid_provenance) == 0,
        "No oversized chunks": len(oversized) == 0,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status:5} | {check}")

    print("=" * 80)


if __name__ == "__main__":
    main()