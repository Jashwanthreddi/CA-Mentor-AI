from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline


pdf_path = Path(
    "backend/sample_data/icai/foundation/paper_1/module_1/chapter_1/"
    "fpp1m1c1u1.pdf"
)

pipeline = IngestionPipeline()

report = pipeline.run(pdf_path)

print("\n=== INGESTION RESULT ===")
print(report.model_dump())