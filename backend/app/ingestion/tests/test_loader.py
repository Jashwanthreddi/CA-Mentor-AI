from pathlib import Path

import pytest

from app.core.exceptions import AppException
from app.ingestion.loader import PDFLoader


def test_load_valid_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    loader = PDFLoader()

    result = loader.load(pdf_path)

    assert result == pdf_path


def test_load_missing_file() -> None:
    loader = PDFLoader()

    pdf_path = Path("does_not_exist.pdf")

    with pytest.raises(AppException) as exc_info:
        loader.load(pdf_path)

    assert exc_info.value.status_code == 404


def test_load_directory(tmp_path: Path) -> None:
    loader = PDFLoader()

    directory = tmp_path / "document.pdf"
    directory.mkdir()

    with pytest.raises(AppException) as exc_info:
        loader.load(directory)

    assert exc_info.value.status_code == 400


def test_load_unsupported_file(tmp_path: Path) -> None:
    loader = PDFLoader()

    txt_path = tmp_path / "document.txt"
    txt_path.write_text("This is not a PDF.")

    with pytest.raises(AppException) as exc_info:
        loader.load(txt_path)

    assert exc_info.value.status_code == 400


def test_pdf_extension_is_case_insensitive(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.PDF"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    loader = PDFLoader()

    result = loader.load(pdf_path)

    assert result == pdf_path