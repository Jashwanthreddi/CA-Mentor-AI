from collections import Counter, defaultdict

from qdrant_client import QdrantClient

from app.core.config import get_settings


def main():
    settings = get_settings()

    # ---------------------------------------------------------
    # Connect to Qdrant
    # ---------------------------------------------------------
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )

    collection_name = settings.QDRANT_COLLECTION

    print("\n" + "=" * 70)
    print("QDRANT INGESTION AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Collection information
    # ---------------------------------------------------------
    info = client.get_collection(collection_name)

    print("\nCOLLECTION")
    print("-" * 40)
    print(f"Collection: {collection_name}")
    print(f"Status: {info.status}")
    print(f"Points: {info.points_count}")
    print(f"Vector configuration: {info.config.params.vectors}")

    # ---------------------------------------------------------
    # 2. Read points
    # ---------------------------------------------------------
    points, next_offset = client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    print(f"\nPoints retrieved: {len(points)}")

    if not points:
        print("\n❌ No points found.")
        return

    # ---------------------------------------------------------
    # 3. Metadata distributions
    # ---------------------------------------------------------
    documents = Counter()
    levels = Counter()
    papers = Counter()
    sources = Counter()
    editions = Counter()
    modules = Counter()
    chapters = Counter()
    units = Counter()
    topics = Counter()
    subtopics = Counter()
    educational_types = Counter()
    versions = Counter()

    page_ranges = []
    chunk_lengths = []

    missing_metadata = defaultdict(int)

    required_fields = [
        "document_id",
        "source",
        "edition",
        "level",
        "paper",
        "educational_type",
        "page_start",
        "page_end",
        "version",
    ]

    for point in points:
        payload = point.payload or {}

        documents[payload.get("document_id")] += 1
        levels[payload.get("level")] += 1
        papers[payload.get("paper")] += 1
        sources[payload.get("source")] += 1
        editions[payload.get("edition")] += 1
        modules[payload.get("module")] += 1
        chapters[payload.get("chapter")] += 1
        units[payload.get("unit")] += 1
        topics[payload.get("topic")] += 1
        subtopics[payload.get("subtopic")] += 1
        educational_types[payload.get("educational_type")] += 1
        versions[payload.get("version")] += 1

        page_start = payload.get("page_start")
        page_end = payload.get("page_end")

        if page_start is not None and page_end is not None:
            page_ranges.append((page_start, page_end))

        text = payload.get("text", "")

        if text:
            chunk_lengths.append(len(text))

        for field in required_fields:
            value = payload.get(field)

            if value is None or value == "":
                missing_metadata[field] += 1

    # ---------------------------------------------------------
    # Helper
    # ---------------------------------------------------------
    def print_counter(title, counter):
        print(f"\n{title}")
        print("-" * 40)

        if not counter:
            print("(none)")
            return

        for key, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], str(item[0]))
        ):
            print(f"{key}: {count}")

    # ---------------------------------------------------------
    # 4. Metadata distributions
    # ---------------------------------------------------------
    print_counter("DOCUMENTS", documents)
    print_counter("LEVELS", levels)
    print_counter("PAPERS", papers)
    print_counter("SOURCES", sources)
    print_counter("EDITIONS", editions)
    print_counter("MODULES", modules)
    print_counter("CHAPTERS", chapters)
    print_counter("UNITS", units)
    print_counter("TOPICS", topics)
    print_counter("SUBTOPICS", subtopics)
    print_counter("EDUCATIONAL TYPES", educational_types)
    print_counter("VERSIONS", versions)

    # ---------------------------------------------------------
    # 5. Missing metadata
    # ---------------------------------------------------------
    print("\nMISSING REQUIRED METADATA")
    print("-" * 40)

    if not missing_metadata:
        print("✅ No required metadata is missing.")
    else:
        for field, count in sorted(missing_metadata.items()):
            print(f"{field}: {count}")

    # ---------------------------------------------------------
    # 6. Page coverage
    # ---------------------------------------------------------
    print("\nPAGE COVERAGE")
    print("-" * 40)

    if page_ranges:
        min_page = min(start for start, _ in page_ranges)
        max_page = max(end for _, end in page_ranges)

        print(f"Minimum page: {min_page}")
        print(f"Maximum page: {max_page}")

        print("\nPage ranges:")
        for start, end in sorted(page_ranges):
            print(f"  {start} - {end}")
    else:
        print("❌ No page information found.")

    # ---------------------------------------------------------
    # 7. Chunk size analysis
    # ---------------------------------------------------------
    print("\nCHUNK SIZE ANALYSIS")
    print("-" * 40)

    if chunk_lengths:
        minimum = min(chunk_lengths)
        maximum = max(chunk_lengths)
        average = sum(chunk_lengths) / len(chunk_lengths)

        print(f"Minimum characters: {minimum}")
        print(f"Maximum characters: {maximum}")
        print(f"Average characters: {average:.2f}")

        short_chunks = [
            length for length in chunk_lengths
            if length < 100
        ]

        very_short_chunks = [
            length for length in chunk_lengths
            if length < 50
        ]

        long_chunks = [
            length for length in chunk_lengths
            if length > 5000
        ]

        print(f"Chunks < 50 chars: {len(very_short_chunks)}")
        print(f"Chunks < 100 chars: {len(short_chunks)}")
        print(f"Chunks > 5000 chars: {len(long_chunks)}")
    else:
        print("❌ No chunk text found.")

    # ---------------------------------------------------------
    # 8. Point ID check
    # ---------------------------------------------------------
    print("\nPOINT ID CHECK")
    print("-" * 40)

    ids = [str(point.id) for point in points]

    if len(ids) == len(set(ids)):
        print("✅ No duplicate point IDs in retrieved points.")
    else:
        print("❌ Duplicate point IDs detected.")

    # ---------------------------------------------------------
    # 9. Show sample points
    # ---------------------------------------------------------
    print("\nSAMPLE POINTS")
    print("-" * 40)

    for index, point in enumerate(points[:5], start=1):
        payload = point.payload or {}

        print(f"\n--- Point {index} ---")

        print(f"ID: {point.id}")
        print(f"Document: {payload.get('document_id')}")
        print(f"Source: {payload.get('source')}")
        print(f"Edition: {payload.get('edition')}")
        print(f"Level: {payload.get('level')}")
        print(f"Paper: {payload.get('paper')}")
        print(f"Module: {payload.get('module')}")
        print(f"Chapter: {payload.get('chapter')}")
        print(f"Unit: {payload.get('unit')}")
        print(f"Topic: {payload.get('topic')}")
        print(f"Subtopic: {payload.get('subtopic')}")
        print(f"Type: {payload.get('educational_type')}")
        print(f"Pages: {payload.get('page_start')} - {payload.get('page_end')}")
        print(f"Version: {payload.get('version')}")

        text = payload.get("text", "")

        if text:
            print(f"Text: {text[:300]!r}")
        else:
            print("Text: <missing>")

    # ---------------------------------------------------------
    # 10. Document-level summary
    # ---------------------------------------------------------
    print("\nDOCUMENT SUMMARY")
    print("-" * 40)

    for document_id, count in documents.items():
        print(f"{document_id}: {count} chunks")

    # ---------------------------------------------------------
    # 11. Expected ingestion check
    # ---------------------------------------------------------
    print("\nEXPECTED INGESTION CHECK")
    print("-" * 40)

    expected_document_id = "doc-14e8f62b76e272d36e50"

    actual_document_count = documents.get(expected_document_id, 0)

    print(f"Expected document: {expected_document_id}")
    print(f"Chunks for document: {actual_document_count}")

    if actual_document_count == 38:
        print("✅ Latest ingestion contains exactly 38 chunks.")
    elif actual_document_count == 0:
        print("⚠️ Latest document ID was not found.")
    else:
        print(
            f"⚠️ Expected 38 chunks for latest document, "
            f"but found {actual_document_count}."
        )

    # ---------------------------------------------------------
    # 12. Finish
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()