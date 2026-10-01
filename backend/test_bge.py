from app.ingestion.embedder import DocumentEmbedder


def main():
    print("Loading BGE-base-en-v1.5...")

    embedder = DocumentEmbedder(
        batch_size=2,
        device="cpu",
    )

    texts = [
        "Accounting is the process of recording financial transactions.",
        "Depreciation is the systematic allocation of the depreciable amount of an asset.",
    ]

    embeddings = embedder.model.encode(
        texts,
        batch_size=2,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"First 5 values: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()