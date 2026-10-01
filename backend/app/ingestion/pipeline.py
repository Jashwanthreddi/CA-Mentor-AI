"""
End-to-end ingestion pipeline for CA Mentor AI.

Pipeline:

PDF
    ↓
Loader
    ↓
Extractor
    ↓
Classifier
    ↓
Parser
    ↓
Chunker
    ↓
Metadata Builder
    ↓
Validator
    ↓
Embedder
    ↓
Qdrant Uploader

The pipeline is responsible only for orchestration.
Each ingestion module owns its own transformation logic.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from app.core.logging import logger
from app.ingestion.classifier import EducationalClassifier
from app.ingestion.chunker import SemanticChunker
from app.ingestion.embedder import DocumentEmbedder
from app.ingestion.extractor import PDFExtractor
from app.ingestion.loader import PDFLoader
from app.ingestion.metadata import MetadataBuilder
from app.ingestion.models import IngestionReport
from app.ingestion.parser import EducationalParser
from app.ingestion.uploader import QdrantUploader
from app.ingestion.validator import DocumentChunkValidator


class IngestionPipeline:
    """
    Orchestrates the complete CA Mentor AI ingestion workflow.

    The pipeline does not implement the individual transformations.
    It coordinates the existing ingestion components.
    """

    def __init__(
        self,
        loader: PDFLoader | None = None,
        extractor: PDFExtractor | None = None,
        classifier: EducationalClassifier | None = None,
        parser: EducationalParser | None = None,
        chunker: SemanticChunker | None = None,
        metadata_builder: MetadataBuilder | None = None,
        validator: DocumentChunkValidator | None = None,
        embedder: DocumentEmbedder | None = None,
        uploader: QdrantUploader | None = None,
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Every component supports dependency injection so the pipeline
        can be tested without requiring real PDF extraction, BGE
        inference, or Qdrant Cloud.
        """

        self.loader = loader or PDFLoader()
        self.extractor = extractor or PDFExtractor()
        self.classifier = classifier or EducationalClassifier()
        self.parser = parser or EducationalParser()
        self.chunker = chunker or SemanticChunker()
        self.metadata_builder = (
            metadata_builder or MetadataBuilder()
        )
        self.validator = (
            validator or DocumentChunkValidator()
        )
        self.embedder = embedder or DocumentEmbedder()
        self.uploader = uploader or QdrantUploader()

    def run(
        self,
        file_path: Path,
        *,
        page_range: tuple[int, int] | None = None,
    ) -> IngestionReport:
        """
        Run the complete ingestion pipeline.

        Parameters
        ----------
        file_path:
            Path to the ICAI PDF.

        page_range:
            Optional inclusive page range for controlled ingestion.
            Example: (1, 3) processes pages 1 through 3.

        Returns
        -------
        IngestionReport
            Summary of the completed ingestion operation.
        """

        pipeline_start = perf_counter()

        logger.info(
            f"Starting ingestion pipeline: {file_path}"
        )

        # ---------------------------------------------------------
        # 1. LOAD
        # ---------------------------------------------------------
        document_path = self.loader.load(file_path)

        logger.info(
            f"[1/10] Document loaded: {document_path}"
        )

        # ---------------------------------------------------------
        # 2. EXTRACT
        # ---------------------------------------------------------
        raw_document = self.extractor.extract(
            document_path,
            page_range=page_range,
        )

        logger.info(
            f"[2/10] Extraction completed: "
            f"{len(raw_document.pages)} pages"
        )

        # ---------------------------------------------------------
        # 3. CLASSIFY
        # ---------------------------------------------------------
        educational_elements = self.classifier.classify(
            raw_document
        )

        logger.info(
            f"[3/10] Classification completed: "
            f"{len(educational_elements)} elements"
        )

        # ---------------------------------------------------------
        # 4. PARSE
        # ---------------------------------------------------------
        knowledge_units = self.parser.parse(
            educational_elements
        )

        logger.info(
            f"[4/10] Parsing completed: "
            f"{len(knowledge_units)} knowledge units"
        )

        # ---------------------------------------------------------
        # 5. CHUNK
        # ---------------------------------------------------------
        chunks = self.chunker.chunk(
            knowledge_units
        )

        logger.info(
            f"[5/10] Chunking completed: "
            f"{len(chunks)} chunks"
        )

        # ---------------------------------------------------------
        # 6. METADATA
        # ---------------------------------------------------------
        chunks = self.metadata_builder.build(chunks)

        logger.info(
            "[6/10] Metadata enrichment completed."
        )

        # ---------------------------------------------------------
        # 7. VALIDATE
        # ---------------------------------------------------------
        validation_result = self.validator.validate(chunks)

        logger.info(
            f"[7/10] Validation completed: "
            f"{validation_result.valid_count} valid, "
            f"{validation_result.invalid_count} invalid"
        )

        if not validation_result.is_valid:
            error_summary = "; ".join(
                validation_result.errors[:10]
            )

            raise ValueError(
                "Ingestion validation failed. "
                f"Invalid chunks: "
                f"{validation_result.invalid_count}. "
                f"Errors: {error_summary}"
            )

        valid_chunks = validation_result.valid_chunks

        # ---------------------------------------------------------
        # 8. EMBED
        # ---------------------------------------------------------
        embedding_start = perf_counter()

        embedded_chunks = self.embedder.embed(
            valid_chunks
        )

        embedding_time = (
            perf_counter() - embedding_start
        )

        logger.info(
            f"[8/10] Embedding completed: "
            f"{len(embedded_chunks)} vectors "
            f"in {embedding_time:.3f}s"
        )

        # ---------------------------------------------------------
        # 9. UPLOAD
        # ---------------------------------------------------------
        tables = sum(
            1
            for chunk in valid_chunks
            if chunk.metadata.has_table
        )

        images = sum(
            1
            for chunk in valid_chunks
            if chunk.metadata.has_image
        )

        upload_report = self.uploader.upload(
            embedded_chunks,
            filename=document_path.name,
            total_pages=raw_document.metadata.pages,
            sections=len(educational_elements),
            eku_count=len(knowledge_units),
            tables=tables,
            images=images,
            embedding_time=embedding_time,
        )

        logger.info(
            f"[9/10] Qdrant upload completed: "
            f"{upload_report.chunks} chunks"
        )

        # ---------------------------------------------------------
        # 10. COMPLETE
        # ---------------------------------------------------------
        total_time = perf_counter() - pipeline_start

        logger.info(
            f"[10/10] Ingestion pipeline completed "
            f"in {total_time:.3f}s."
        )

        return upload_report