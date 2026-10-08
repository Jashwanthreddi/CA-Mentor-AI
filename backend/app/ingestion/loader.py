"""
PDF Loader

Purpose
-------
Validate and return a supported PDF document.

The loader is intentionally lightweight.

It does NOT:
    - parse PDF content
    - extract text
    - run OCR
    - inspect tables
    - create educational elements
    - create chunks
    - create embeddings

Those responsibilities belong to the extraction and ingestion layers.
"""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import AppException
from loguru import logger


class PDFLoader:
    """Validate and return a supported PDF document."""

    SUPPORTED_EXTENSIONS = {".pdf"}
    PDF_SIGNATURE = b"%PDF-"

    def load(self, file_path: Path) -> Path:
        """
        Validate the supplied PDF path.

        Parameters
        ----------
        file_path:
            Path to the PDF document.

        Returns
        -------
        Path
            Validated PDF path.

        Raises
        ------
        AppException
            If the path does not exist, is not a file,
            is empty, has an unsupported extension,
            or does not have a valid PDF signature.
        """

        logger.info(
            f"Validating document: {file_path}"
        )

        # --------------------------------------------------------------
        # Path validation
        # --------------------------------------------------------------

        if not file_path.exists():
            raise AppException(
                message=f"File not found: {file_path}",
                status_code=404,
            )

        if not file_path.is_file():
            raise AppException(
                message=f"Not a valid file: {file_path}",
                status_code=400,
            )

        # --------------------------------------------------------------
        # Extension validation
        # --------------------------------------------------------------

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise AppException(
                message=(
                    f"Unsupported file type: "
                    f"{file_path.suffix}"
                ),
                status_code=400,
            )

        # --------------------------------------------------------------
        # File size validation
        # --------------------------------------------------------------

        file_size = file_path.stat().st_size

        if file_size == 0:
            raise AppException(
                message="PDF file is empty.",
                status_code=400,
            )

        # --------------------------------------------------------------
        # PDF signature validation
        # --------------------------------------------------------------

        try:
            with file_path.open("rb") as file:
                signature = file.read(5)

        except OSError as exc:
            raise AppException(
                message=(
                    f"Unable to read PDF file: "
                    f"{file_path}"
                ),
                status_code=400,
            ) from exc

        if signature != self.PDF_SIGNATURE:
            raise AppException(
                message=(
                    "File does not contain a valid PDF signature."
                ),
                status_code=400,
            )

        logger.info(
            "Document validation completed successfully | "
            f"file={file_path.name} | "
            f"size={file_size} bytes"
        )

        return file_path