from app.rag.bm25_corpus import BM25CorpusLoader


def main() -> None:
    chunks = BM25CorpusLoader().load()

    print("=" * 100)
    print(f"TOTAL CHUNKS: {len(chunks)}")
    print("=" * 100)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n[{index}]")
        print(f"Chunk ID       : {chunk.chunk_id}")
        print(f"Pages          : {chunk.page_start}-{chunk.page_end}")
        print(f"Type           : {chunk.educational_type}")
        print(f"Chapter        : {chunk.chapter}")
        print(f"Unit           : {chunk.unit}")
        print(f"Topic          : {chunk.topic}")
        print(f"Text           : {chunk.text[:500].replace(chr(10), ' ')}")
        print("-" * 100)


if __name__ == "__main__":
    main()