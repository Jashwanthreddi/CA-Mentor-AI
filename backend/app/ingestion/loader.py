"""
PDF Loader

Purpose
-------
Validate and return a supported PDF document.

The loader is intentionally lightweight. It does not parse or
extract document content. That responsibility belongs to the
extraction layer.
"""

from pathlib import Path

from app.core.exceptions import AppException
from app.core.logging import logger


class PDFLoader:
    """Validate and return a supported PDF document."""

    SUPPORTED_EXTENSIONS = {".pdf"}

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
            The validated PDF path.

        Raises
        ------
        AppException
            If the path does not exist, is not a file,
            or has an unsupported file extension.
        """
        logger.info(f"Validating document: {file_path}")

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

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise AppException(
                message=f"Unsupported file type: {file_path.suffix}",
                status_code=400,
            )

        logger.info("Document validation completed successfully.")

        return file_path