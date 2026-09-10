"""
PDF Loader

Purpose
-------
Validate and return a supported PDF document.

Input
-----
Path

Output
------
Validated Path

Responsibilities
----------------
- Validate file existence
- Validate supported extension
- Validate file accessibility

Must NOT
--------
- Read PDF contents
- Parse PDF
- Call Docling
- Generate metadata

Consumers
---------
Docling Extractor
"""

from pathlib import Path
from app.core.logging import logger
from app.core.exceptions import AppException

class PDFLoader:
    # responsible for validating pdf documnets before extraction
    SUPPORTED_EXTENSIONS={".pdf"}

    def load(self,file_path:path) ->Path:
        logger.info(f"Validating document:{file_path}")

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
                message=(
                    f"Unsupported file type:"
                    f"{file_path.suffix}"
                ),
                status_code=400,
            )
        logger.info("Document validation completed successfully.")

        return file_path