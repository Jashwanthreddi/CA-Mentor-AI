"""
PDF extraction boundary for CA Mentor AI.

Responsibilities
----------------
Convert ICAI digital PDF documents into the application's
RawDocument model.

Pipeline boundary:

    PDF
     ↓
    Docling
     ↓
    RawDocument
     ↓
    Classifier

Design goals
------------
1. Support different PDF page counts.
2. Process documents in bounded page ranges.
3. Recover from problematic page ranges.
4. Preserve page numbers and source provenance.
5. Preserve text, tables, images, formulas, lists, headers,
   footers, captions, and footnotes.
6. Avoid eagerly loading image bytes.
7. Keep Docling-specific implementation isolated here.
8. Produce explicit extraction diagnostics.
9. Preserve incomplete-page information instead of silently
   treating failed extraction as successful.
10. Keep V1 optimized for ICAI digital PDFs.

Important
---------
V1 intentionally targets digital ICAI PDFs.

OCR is disabled by default.

Educational interpretation belongs to classifier.py.
Educational grouping belongs to parser.py.
Chunking belongs to chunker.py.
Embedding belongs to embedder.py.
Qdrant persistence belongs to uploader.py.
"""

from __future__ import annotations

import gc

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid5, NAMESPACE_URL

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc import (
    DocItem,
    DoclingDocument,
)

from app.core.exceptions import AppException
from app.ingestion.raw_document import (
    BlockType,
    BoundingBox,
    RawBlock,
    RawDocument,
    RawDocumentMetadata,
    RawPage,
)

from loguru import logger


class PDFExtractor:
    """
    Convert an ICAI digital PDF into RawDocument.

    The extractor performs structural extraction only.

    It does NOT:
        - classify educational meaning
        - create EKUs
        - chunk text
        - create embeddings
        - upload to Qdrant
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    DEFAULT_OCR = False

    DEFAULT_TABLE_STRUCTURE = False

    DEFAULT_PAGE_BATCH_SIZE = 5

    DEFAULT_RETRY_FAILED_PAGES = True

    DEFAULT_MAX_PAGE_ATTEMPTS = 2

    MAX_RECURSIVE_SPLIT_DEPTH = 3

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        do_ocr: bool = DEFAULT_OCR,
        do_table_structure: bool = DEFAULT_TABLE_STRUCTURE,
        page_batch_size: int = DEFAULT_PAGE_BATCH_SIZE,
        retry_failed_pages: bool = DEFAULT_RETRY_FAILED_PAGES,
        max_page_attempts: int = DEFAULT_MAX_PAGE_ATTEMPTS,
    ) -> None:

        if page_batch_size < 1:
            raise ValueError(
                "page_batch_size must be at least 1."
            )

        if max_page_attempts < 1:
            raise ValueError(
                "max_page_attempts must be at least 1."
            )

        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        self.page_batch_size = page_batch_size
        self.retry_failed_pages = retry_failed_pages
        self.max_page_attempts = max_page_attempts

        logger.info(
            "PDFExtractor initialized | "
            f"OCR={self.do_ocr} | "
            f"table_structure={self.do_table_structure} | "
            f"batch_size={self.page_batch_size} | "
            f"retry={self.retry_failed_pages}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        file_path: Path,
        *,
        page_range: tuple[int, int] | None = None,
    ) -> RawDocument:

        start_time = perf_counter()

        logger.info(
            f"Starting PDF extraction: {file_path}"
        )

        self._validate_input(file_path)

        # --------------------------------------------------------------
        # Determine requested range
        # --------------------------------------------------------------

        if page_range is not None:

            start_page, end_page = self._validate_requested_range(
                page_range
            )

            logger.info(
                "Requested page range: "
                f"{start_page}-{end_page}"
            )

        else:
            start_page = 1
            end_page = None

        pages_by_number: dict[int, RawPage] = {}

        failed_pages: list[int] = []

        warnings: list[str] = []

        # --------------------------------------------------------------
        # Explicit requested range
        # --------------------------------------------------------------

        if end_page is not None:

            batch_ranges = self._build_page_batches(
                start_page=start_page,
                end_page=end_page,
            )

            for batch_start, batch_end in batch_ranges:

                self._process_range(
                    file_path=file_path,
                    start_page=batch_start,
                    end_page=batch_end,
                    pages_by_number=pages_by_number,
                    failed_pages=failed_pages,
                    warnings=warnings,
                )

                gc.collect()

        # --------------------------------------------------------------
        # Full-document extraction
        # --------------------------------------------------------------

        else:

            self._extract_full_document(
                file_path=file_path,
                pages_by_number=pages_by_number,
                failed_pages=failed_pages,
                warnings=warnings,
            )

            if not pages_by_number:
                raise AppException(
                    message=(
                        "PDF extraction failed. "
                        "No pages could be extracted."
                    ),
                    status_code=422,
                )

            start_page = 1

            # IMPORTANT:
            # Use the actual PDF page count rather than the number
            # of successfully extracted pages.
            end_page = self._get_pdf_page_count(file_path)

        # --------------------------------------------------------------
        # Build RawDocument
        # --------------------------------------------------------------

        raw_document = self._build_raw_document(
            file_path=file_path,
            pages_by_number=pages_by_number,
            requested_start_page=start_page,
            requested_end_page=end_page,
            failed_pages=failed_pages,
            warnings=warnings,
        )

        elapsed = perf_counter() - start_time

        self._log_extraction_statistics(
            raw_document=raw_document,
            elapsed=elapsed,
        )

        extracted_pages = len(raw_document.pages)

        if extracted_pages == 0:
            raise AppException(
                message="Extraction produced zero usable pages.",
                status_code=422,
            )

        if failed_pages:
            logger.error(
                "PDF extraction completed with failed pages | "
                f"pages={sorted(set(failed_pages))}"
            )

        logger.info(
            "PDF extraction completed | "
            f"requested_range={start_page}-{end_page} | "
            f"extracted_pages={extracted_pages} | "
            f"failed_pages={len(set(failed_pages))} | "
            f"elapsed={elapsed:.2f}s"
        )

        return raw_document

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_input(
        self,
        file_path: Path,
    ) -> None:

        if not file_path.exists():
            raise AppException(
                message=f"PDF file not found: {file_path}",
                status_code=404,
            )

        if not file_path.is_file():
            raise AppException(
                message=f"Path is not a file: {file_path}",
                status_code=400,
            )

        if file_path.suffix.lower() != ".pdf":
            raise AppException(
                message=(
                    f"Unsupported file type: "
                    f"{file_path.suffix}"
                ),
                status_code=400,
            )

        if file_path.stat().st_size == 0:
            raise AppException(
                message="PDF file is empty.",
                status_code=400,
            )

        try:
            with file_path.open("rb") as file:
                signature = file.read(5)

        except OSError as exc:
            raise AppException(
                message=(
                    f"Unable to read PDF file: {file_path}"
                ),
                status_code=400,
            ) from exc

        if signature != b"%PDF-":
            raise AppException(
                message="File does not contain a valid PDF signature.",
                status_code=400,
            )

    # ------------------------------------------------------------------
    # Requested page range
    # ------------------------------------------------------------------

    def _validate_requested_range(
        self,
        page_range: tuple[int, int],
    ) -> tuple[int, int]:

        if len(page_range) != 2:
            raise ValueError(
                "page_range must contain exactly two values."
            )

        start_page, end_page = page_range

        if start_page < 1:
            raise ValueError(
                "page_range start page must be >= 1."
            )

        if end_page < start_page:
            raise ValueError(
                "page_range end page must be >= start page."
            )

        return start_page, end_page

    # ------------------------------------------------------------------
    # Actual PDF page count
    # ------------------------------------------------------------------

    def _get_pdf_page_count(
        self,
        file_path: Path,
    ) -> int:
        """
        Determine the actual number of pages in the PDF.

        This is used ONLY for page-count discovery.

        Docling remains responsible for actual content extraction.
        """

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))

            page_count = len(reader.pages)

            if page_count < 1:
                raise ValueError(
                    "PDF contains zero pages."
                )

            logger.info(
                "Actual PDF page count discovered | "
                f"pages={page_count}"
            )

            return page_count

        except Exception as exc:
            logger.error(
                "Unable to determine PDF page count | "
                f"file={file_path} | "
                f"error={exc}"
            )

            raise AppException(
                message=(
                    "Unable to determine the total number "
                    "of pages in the PDF."
                ),
                status_code=422,
            ) from exc

    # ------------------------------------------------------------------
    # Full-document adaptive extraction
    # ------------------------------------------------------------------

    def _extract_full_document(
        self,
        *,
        file_path: Path,
        pages_by_number: dict[int, RawPage],
        failed_pages: list[int],
        warnings: list[str],
    ) -> None:
        """
        Extract the complete PDF.

        IMPORTANT:
        Do NOT use Docling's returned page count to determine EOF.

        Docling can return fewer pages because a particular page
        failed during preprocessing/conversion.

        The real PDF page count is determined independently.
        """

        total_pages = self._get_pdf_page_count(file_path)

        logger.info(
            "Starting complete PDF extraction | "
            f"total_pages={total_pages}"
        )

        current_page = 1

        while current_page <= total_pages:

            batch_end = min(
                current_page + self.page_batch_size - 1,
                total_pages,
            )

            logger.info(
                "Processing adaptive batch | "
                f"pages={current_page}-{batch_end} | "
                f"total_pages={total_pages}"
            )

            self._process_range(
                file_path=file_path,
                start_page=current_page,
                end_page=batch_end,
                pages_by_number=pages_by_number,
                failed_pages=failed_pages,
                warnings=warnings,
            )

            current_page = batch_end + 1

            # Release Python/native objects between conversions.
            gc.collect()

        logger.info(
            "Adaptive extraction completed | "
            f"requested_pages=1-{total_pages} | "
            f"extracted_pages={len(pages_by_number)} | "
            f"failed_pages={sorted(set(failed_pages))}"
        )

    # ------------------------------------------------------------------
    # Range processing
    # ------------------------------------------------------------------


    def _process_range(
        self,
        *,
        file_path: Path,
        start_page: int,
        end_page: int,
        pages_by_number: dict[int, RawPage],
        failed_pages: list[int],
        warnings: list[str],
        split_depth: int = 0,
    ) -> None:

        logger.info(
            "Extracting range | "
            f"{start_page}-{end_page}"
        )

        document = self._extract_range(
            file_path=file_path,
            start_page=start_page,
            end_page=end_page,
        )

        if document is not None:

            returned_pages = self._get_document_page_numbers(
                document
            )

            expected_pages = set(
                range(start_page, end_page + 1)
            )

            missing_pages = expected_pages - returned_pages

            # --------------------------------------------------------------
            # Collect everything Docling returned.
            # --------------------------------------------------------------

            self._collect_pages_from_document(
                document=document,
                pages_by_number=pages_by_number,
            )

            # --------------------------------------------------------------
            # Detect pages that technically exist in Docling's response
            # but contain ZERO extracted blocks.
            #
            # This is important because previously such pages were treated
            # as successful extraction.
            # --------------------------------------------------------------

            empty_pages = {
                page_number
                for page_number in returned_pages
                if (
                    page_number in pages_by_number
                    and not pages_by_number[page_number].blocks
                )
            }

            # --------------------------------------------------------------
            # Retry both:
            #
            # 1. Pages completely missing from Docling output
            # 2. Pages returned by Docling but containing zero blocks
            # --------------------------------------------------------------

            pages_to_retry = (
                missing_pages | empty_pages
            )

            if pages_to_retry:

                if missing_pages:

                    logger.warning(
                        "Docling returned incomplete page coverage | "
                        f"requested={start_page}-{end_page} | "
                        f"returned={sorted(returned_pages)} | "
                        f"missing={sorted(missing_pages)}"
                    )

                    warning = (
                        "Incomplete Docling page coverage for "
                        f"{start_page}-{end_page}. "
                        f"Missing pages={sorted(missing_pages)}"
                    )

                    if warning not in warnings:
                        warnings.append(warning)

                if empty_pages:

                    logger.warning(
                        "Docling returned pages with no extracted blocks | "
                        f"pages={sorted(empty_pages)}"
                    )

                    warning = (
                        "Docling returned empty pages for "
                        f"{start_page}-{end_page}. "
                        f"Pages={sorted(empty_pages)}"
                    )

                    if warning not in warnings:
                        warnings.append(warning)

                logger.info(
                    "Retrying problematic pages individually | "
                    f"pages={sorted(pages_to_retry)}"
                )

                for page_number in sorted(pages_to_retry):

                    self._process_single_page(
                        file_path=file_path,
                        page_number=page_number,
                        pages_by_number=pages_by_number,
                        failed_pages=failed_pages,
                        warnings=warnings,
                    )

            return

        # --------------------------------------------------------------
        # Complete range failure
        # --------------------------------------------------------------

        if (
            start_page != end_page
            and split_depth < self.MAX_RECURSIVE_SPLIT_DEPTH
        ):

            middle = (
                start_page + end_page
            ) // 2

            logger.warning(
                "Splitting failed extraction range | "
                f"{start_page}-{end_page} -> "
                f"{start_page}-{middle} and "
                f"{middle + 1}-{end_page}"
            )

            self._process_range(
                file_path=file_path,
                start_page=start_page,
                end_page=middle,
                pages_by_number=pages_by_number,
                failed_pages=failed_pages,
                warnings=warnings,
                split_depth=split_depth + 1,
            )

            self._process_range(
                file_path=file_path,
                start_page=middle + 1,
                end_page=end_page,
                pages_by_number=pages_by_number,
                failed_pages=failed_pages,
                warnings=warnings,
                split_depth=split_depth + 1,
            )

            return

        # --------------------------------------------------------------
        # Single page / final failure
        # --------------------------------------------------------------

        for page_number in range(
            start_page,
            end_page + 1,
        ):

            self._process_single_page(
                file_path=file_path,
                page_number=page_number,
                pages_by_number=pages_by_number,
                failed_pages=failed_pages,
                warnings=warnings,
            )

    # ------------------------------------------------------------------
    # Single page processing
    # ------------------------------------------------------------------

    def _process_single_page(
        self,
        *,
        file_path: Path,
        page_number: int,
        pages_by_number: dict[int, RawPage],
        failed_pages: list[int],
        warnings: list[str],
    ) -> None:
        """
        Extract one problematic page.

        A page must contain at least one extracted block before it is
        considered successfully extracted.

        Pages that are returned by Docling but contain zero blocks are
        retried individually because Docling can sometimes return the
        page structure without successfully extracting its content.
        """

        logger.info(
            f"Extracting individual page: {page_number}"
        )

        for attempt in range(
            1,
            self.max_page_attempts + 1,
        ):

            logger.info(
                f"Page {page_number} attempt "
                f"{attempt}/{self.max_page_attempts}"
            )

            document = None

            try:

                document = self._extract_range(
                    file_path=file_path,
                    start_page=page_number,
                    end_page=page_number,
                )

                if document is None:

                    logger.warning(
                        f"Page {page_number} returned no document "
                        f"on attempt {attempt}."
                    )

                    continue

                returned_pages = self._get_document_page_numbers(
                    document
                )

                if page_number not in returned_pages:

                    logger.warning(
                        f"Docling did not return page "
                        f"{page_number} on attempt {attempt}."
                    )

                    continue

                # ----------------------------------------------------------
                # Collect the returned page.
                #
                # If the page already exists but is empty, a successful retry
                # will add its blocks to the existing RawPage.
                # ----------------------------------------------------------

                self._collect_pages_from_document(
                    document=document,
                    pages_by_number=pages_by_number,
                )

                page = pages_by_number.get(page_number)

                # ----------------------------------------------------------
                # IMPORTANT:
                #
                # Merely receiving a page from Docling is NOT enough.
                #
                # We require actual extracted blocks.
                # ----------------------------------------------------------

                if page is None:

                    logger.warning(
                        f"Page {page_number} was returned by Docling "
                        f"but no RawPage was created on attempt "
                        f"{attempt}."
                    )

                    continue

                if not page.blocks:

                    logger.warning(
                        f"Page {page_number} returned with ZERO blocks "
                        f"on attempt {attempt}. "
                        f"Retrying."
                    )

                    continue

                logger.info(
                    f"Page {page_number} extracted successfully | "
                    f"blocks={len(page.blocks)}"
                )

                return

            except Exception as exc:

                logger.error(
                    f"Page {page_number} extraction error | "
                    f"attempt={attempt} | "
                    f"error={exc}"
                )

            finally:

                document = None
                gc.collect()

        # --------------------------------------------------------------
        # After all attempts
        # --------------------------------------------------------------

        page = pages_by_number.get(page_number)

        # --------------------------------------------------------------
        # If Docling consistently returned the page but with zero blocks,
        # keep it as an empty page rather than incorrectly reporting a
        # successful extraction.
        #
        # We do not mark it as a hard extraction failure because the page
        # itself was returned successfully by Docling.
        # --------------------------------------------------------------

        if page is not None and not page.blocks:

            warning = (
                f"Page {page_number} remained empty after "
                f"{self.max_page_attempts} attempts."
            )

            if warning not in warnings:
                warnings.append(warning)

            logger.warning(
                f"Page {page_number} remained empty after "
                f"{self.max_page_attempts} attempts. "
                f"Treating it as an empty page."
            )

            return

        # --------------------------------------------------------------
        # True permanent failure:
        # Docling could not return the page at all.
        # --------------------------------------------------------------

        if page_number not in failed_pages:
            failed_pages.append(page_number)

        warning = (
            f"Page {page_number} could not be extracted "
            f"after {self.max_page_attempts} attempts."
        )

        if warning not in warnings:
            warnings.append(warning)

        logger.error(
            f"Page {page_number} failed permanently after "
            f"{self.max_page_attempts} attempts."
        )



    # ------------------------------------------------------------------
    # Docling conversion
    # ------------------------------------------------------------------

    def _extract_range(
        self,
        *,
        file_path: Path,
        start_page: int,
        end_page: int,
    ) -> DoclingDocument | None:

        converter = self._create_converter()

        try:

            result = converter.convert(
                file_path,
                page_range=(
                    start_page,
                    end_page,
                ),
            )

            document = result.document

            if document is None:
                logger.warning(
                    "Docling returned no document | "
                    f"range={start_page}-{end_page}"
                )

                return None

            returned_pages = self._get_document_page_numbers(
                document
            )

            if not returned_pages:
                logger.warning(
                    "Docling returned document with no pages | "
                    f"range={start_page}-{end_page}"
                )

                return None

            logger.info(
                "Docling conversion completed | "
                f"requested={start_page}-{end_page} | "
                f"returned={sorted(returned_pages)}"
            )

            return document

        except Exception as exc:

            logger.error(
                "Docling extraction failed | "
                f"range={start_page}-{end_page} | "
                f"error={exc}"
            )

            return None

        finally:

            # Release converter resources after every conversion.
            del converter

            gc.collect()

    # ------------------------------------------------------------------
    # Converter creation
    # ------------------------------------------------------------------

    def _create_converter(self) -> DocumentConverter:

        pipeline_options = PdfPipelineOptions()

        # V1: ICAI digital PDFs.
        pipeline_options.do_ocr = self.do_ocr

        # Disabled because of memory constraints.
        pipeline_options.do_table_structure = (
            self.do_table_structure
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    # ------------------------------------------------------------------
    # Page batches
    # ------------------------------------------------------------------

    def _build_page_batches(
        self,
        *,
        start_page: int,
        end_page: int,
    ) -> list[tuple[int, int]]:

        batches: list[tuple[int, int]] = []

        current = start_page

        while current <= end_page:

            batch_end = min(
                current + self.page_batch_size - 1,
                end_page,
            )

            batches.append(
                (
                    current,
                    batch_end,
                )
            )

            current = batch_end + 1

        return batches

    # ------------------------------------------------------------------
    # Page discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _get_document_page_numbers(
        document: DoclingDocument,
    ) -> set[int]:

        page_numbers: set[int] = set()

        for page_number in document.pages.keys():

            try:

                page_numbers.add(
                    int(page_number)
                )

            except (TypeError, ValueError):
                continue

        return page_numbers

    # ------------------------------------------------------------------
    # RawDocument construction
    # ------------------------------------------------------------------

    def _build_raw_document(
        self,
        *,
        file_path: Path,
        pages_by_number: dict[int, RawPage],
        requested_start_page: int,
        requested_end_page: int,
        failed_pages: list[int],
        warnings: list[str],
    ) -> RawDocument:

        raw_pages = sorted(
            pages_by_number.values(),
            key=lambda page: page.page_number,
        )

        parser_version = self._get_docling_version()

        # IMPORTANT:
        # Do not use len(raw_pages).
        #
        # len(raw_pages) represents successfully extracted pages,
        # not the actual PDF page count.
        actual_page_count = self._get_pdf_page_count(
            file_path
        )

        metadata = RawDocumentMetadata(
            filename=file_path.name,
            title=file_path.stem,
            pages=actual_page_count,
            language="English",
            parser="docling",
            parser_version=parser_version,
        )

        if failed_pages:

            unique_failed_pages = sorted(
                set(failed_pages)
            )

            logger.error(
                "RawDocument has incomplete extraction coverage | "
                f"failed_pages={unique_failed_pages} | "
                f"extracted_pages={len(raw_pages)} | "
                f"expected_pages={actual_page_count}"
            )

            warning = (
                "Incomplete PDF extraction. "
                f"Failed pages={unique_failed_pages}. "
                f"Extracted={len(raw_pages)}/"
                f"{actual_page_count} pages."
            )

            if warning not in warnings:
                warnings.append(warning)

        return RawDocument(
            metadata=metadata,
            pages=raw_pages,
        )

    # ------------------------------------------------------------------
    # Collect Docling pages
    # ------------------------------------------------------------------

    def _collect_pages_from_document(
        self,
        *,
        document: DoclingDocument,
        pages_by_number: dict[int, RawPage],
    ) -> None:

        # --------------------------------------------------------------
        # Create RawPage objects.
        # --------------------------------------------------------------

        for page_number, page in document.pages.items():

            page_number_int = int(page_number)

            width, height = self._get_page_dimensions(
                page=page
            )

            if page_number_int not in pages_by_number:

                pages_by_number[page_number_int] = RawPage(
                    page_number=page_number_int,
                    width=width,
                    height=height,
                    blocks=[],
                )

        # --------------------------------------------------------------
        # Convert reading-order items.
        # --------------------------------------------------------------

        for item, _level in document.iterate_items():

            self._convert_item(
                item=item,
                document=document,
                pages_by_number=pages_by_number,
            )

        # --------------------------------------------------------------
        # Sort blocks by page/provenance where possible.
        # --------------------------------------------------------------

        for page in pages_by_number.values():

            page.blocks.sort(
                key=lambda block: (
                    block.bbox.top
                    if block.bbox is not None
                    else float("inf"),
                    block.bbox.left
                    if block.bbox is not None
                    else float("inf"),
                )
            )

    # ------------------------------------------------------------------
    # Item conversion
    # ------------------------------------------------------------------

    def _convert_item(
        self,
        *,
        item: DocItem,
        document: DoclingDocument,
        pages_by_number: dict[int, RawPage],
    ) -> None:

        provenance = getattr(
            item,
            "prov",
            None,
        )

        if not provenance:
            return

        block_type = self._map_block_type(item)

        text = self._extract_text(
            item=item,
            document=document,
        )

        metadata = self._extract_item_metadata(
            item=item,
            document=document,
        )

        for prov in provenance:

            page_number = getattr(
                prov,
                "page_no",
                None,
            )

            if page_number is None:
                continue

            page_number = int(page_number)

            if page_number not in pages_by_number:
                continue

            bbox = self._extract_bbox(
                prov=prov,
                document=document,
                page_number=page_number,
            )

            block_id = self._generate_block_id(
                item=item,
                document=document,
                page_number=page_number,
                prov=prov,
            )

            raw_block = RawBlock(
                id=block_id,
                page=page_number,
                block_type=block_type,
                text=text,
                bbox=bbox,
                metadata=metadata,
            )

            pages_by_number[
                page_number
            ].blocks.append(raw_block)

    # ------------------------------------------------------------------
    # Block type mapping
    # ------------------------------------------------------------------

    def _map_block_type(
        self,
        item: DocItem,
    ) -> BlockType:

        label = self._get_item_label(item)

        mapping = {
            "title": BlockType.HEADING,
            "section_header": BlockType.HEADING,

            "paragraph": BlockType.PARAGRAPH,
            "text": BlockType.PARAGRAPH,

            "table": BlockType.TABLE,

            "picture": BlockType.IMAGE,
            "chart": BlockType.IMAGE,

            "list_item": BlockType.LIST,

            "caption": BlockType.CAPTION,

            "footnote": BlockType.FOOTNOTE,

            "formula": BlockType.FORMULA,

            "page_header": BlockType.HEADER,
            "page_footer": BlockType.FOOTER,
        }

        return mapping.get(
            label,
            BlockType.PARAGRAPH,
        )

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text(
        self,
        *,
        item: DocItem,
        document: DoclingDocument,
    ) -> str:
        """Extract normalized text from a Docling item."""

        label = self._get_item_label(item)

        # --------------------------------------------------------------
        # Tables
        # --------------------------------------------------------------

        if label == "table":

            try:

                markdown = item.export_to_markdown(
                    doc=document,
                )

                if markdown:
                    return self._clean_text(markdown)

            except Exception as exc:

                logger.warning(
                    f"Failed to export table: {exc}"
                )

            return ""

        # --------------------------------------------------------------
        # Images / charts
        # --------------------------------------------------------------

        if label in {
            "picture",
            "chart",
        }:

            caption_method = getattr(
                item,
                "caption_text",
                None,
            )

            if callable(caption_method):

                try:

                    caption = caption_method(document)

                except Exception as exc:

                    logger.debug(
                        "Unable to extract picture caption | "
                        f"error={exc}"
                    )

                    caption = ""

            else:
                caption = caption_method

            if caption:
                return self._clean_text(
                    str(caption)
                )

            return ""

        # --------------------------------------------------------------
        # Generic text
        # --------------------------------------------------------------

        text = getattr(
            item,
            "text",
            "",
        )

        if text is None:
            return ""

        return self._clean_text(
            str(text)
        )

    # ------------------------------------------------------------------
    # Item label
    # ------------------------------------------------------------------

    @staticmethod
    def _get_item_label(
        item: DocItem,
    ) -> str:

        label = getattr(
            item,
            "label",
            "",
        )

        if label is None:
            return ""

        value = getattr(
            label,
            "value",
            label,
        )

        return str(value).lower()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _extract_item_metadata(
        self,
        *,
        item: DocItem,
        document: DoclingDocument,
    ) -> dict[str, Any]:
        """Extract lightweight metadata from a Docling item."""

        metadata: dict[str, Any] = {}

        label = self._get_item_label(item)

        metadata["docling_label"] = label

        # --------------------------------------------------------------
        # Tables
        # --------------------------------------------------------------

        if label == "table":

            try:

                dataframe = item.export_to_dataframe(
                    doc=document,
                )

                if dataframe is not None:

                    metadata["rows"] = int(
                        dataframe.shape[0]
                    )

                    metadata["columns"] = int(
                        dataframe.shape[1]
                    )

            except Exception as exc:

                logger.debug(
                    "Unable to extract table dimensions | "
                    f"error={exc}"
                )

        # --------------------------------------------------------------
        # Images / charts
        # --------------------------------------------------------------

        if label in {
            "picture",
            "chart",
        }:

            caption_method = getattr(
                item,
                "caption_text",
                None,
            )

            if callable(caption_method):

                try:

                    caption = caption_method(document)

                except Exception as exc:

                    logger.debug(
                        "Unable to extract picture caption metadata | "
                        f"error={exc}"
                    )

                    caption = ""

            else:
                caption = caption_method

            if caption:
                metadata["caption"] = self._clean_text(
                    str(caption)
                )

        return metadata

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def _extract_bbox(
        self,
        *,
        prov: Any,
        document: DoclingDocument,
        page_number: int,
    ) -> BoundingBox | None:

        bbox = getattr(
            prov,
            "bbox",
            None,
        )

        if bbox is None:
            return None

        page = document.pages.get(
            page_number
        )

        if page is None:
            return None

        _, page_height = self._get_page_dimensions(
            page=page
        )

        try:

            normalized_bbox = (
                bbox.to_top_left_origin(
                    page_height=page_height
                )
            )

            return BoundingBox(
                left=float(
                    normalized_bbox.l
                ),
                top=float(
                    normalized_bbox.t
                ),
                right=float(
                    normalized_bbox.r
                ),
                bottom=float(
                    normalized_bbox.b
                ),
            )

        except Exception as exc:

            logger.debug(
                "Unable to normalize bounding box | "
                f"page={page_number} | "
                f"error={exc}"
            )

            return None

    # ------------------------------------------------------------------
    # Page dimensions
    # ------------------------------------------------------------------

    @staticmethod
    def _get_page_dimensions(
        *,
        page: Any,
    ) -> tuple[float, float]:

        try:

            size = page.size

            width = float(
                size.width
            )

            height = float(
                size.height
            )

            if width > 0 and height > 0:
                return width, height

        except Exception:
            pass

        # A4 fallback in points.
        return 595.0, 842.0

    # ------------------------------------------------------------------
    # Block ID
    # ------------------------------------------------------------------

    def _generate_block_id(
        self,
        *,
        item: DocItem,
        document: DoclingDocument,
        page_number: int,
        prov: Any,
    ) -> str:

        item_ref = getattr(
            item,
            "self_ref",
            None,
        )

        if item_ref:

            identity = str(item_ref)

        else:

            label = self._get_item_label(item)

            text = self._extract_text(
                item=item,
                document=document,
            )

            identity = f"{label}|{text}"

        bbox = getattr(
            prov,
            "bbox",
            None,
        )

        bbox_text = ""

        if bbox is not None:

            bbox_text = (
                f"{getattr(bbox, 'l', 0)}|"
                f"{getattr(bbox, 't', 0)}|"
                f"{getattr(bbox, 'r', 0)}|"
                f"{getattr(bbox, 'b', 0)}"
            )

        identity_string = (
            f"{page_number}|"
            f"{identity}|"
            f"{bbox_text}"
        )

        block_uuid = uuid5(
            NAMESPACE_URL,
            identity_string,
        )

        return f"block-{block_uuid}"

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        if not text:
            return ""

        return " ".join(
            text.replace(
                "\x00",
                " ",
            ).split()
        )

    # ------------------------------------------------------------------
    # Docling version
    # ------------------------------------------------------------------

    @staticmethod
    def _get_docling_version() -> str:

        try:

            import docling

            return str(
                getattr(
                    docling,
                    "__version__",
                    "unknown",
                )
            )

        except Exception:

            return "unknown"

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _log_extraction_statistics(
        self,
        *,
        raw_document: RawDocument,
        elapsed: float,
    ) -> None:

        block_counter: Counter[str] = Counter()

        total_blocks = 0

        total_characters = 0

        pages_with_content = 0

        empty_pages = 0

        for page in raw_document.pages:

            if page.blocks:
                pages_with_content += 1

            else:
                empty_pages += 1

            total_blocks += len(
                page.blocks
            )

            for block in page.blocks:

                block_counter[
                    block.block_type.value
                ] += 1

                total_characters += len(
                    block.text
                )

        logger.info(
            "PDF extraction statistics | "
            f"pages={len(raw_document.pages)} | "
            f"declared_total_pages={raw_document.metadata.pages} | "
            f"pages_with_content={pages_with_content} | "
            f"empty_pages={empty_pages} | "
            f"blocks={total_blocks} | "
            f"characters={total_characters} | "
            f"block_types={dict(block_counter)} | "
            f"elapsed={elapsed:.2f}s"
        )