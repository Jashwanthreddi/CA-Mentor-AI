"""
PDF Extraction Layer

Purpose
-------
Convert a PDF document into the application's own RawDocument model.

Responsibilities
----------------
- Use Docling for PDF extraction.
- Preserve structural information from the PDF.
- Normalize Docling-specific data into application-owned models.
- Preserve page numbers, bounding boxes, tables, images, and text.
- Keep Docling implementation details isolated inside this module.

Important
---------
This layer performs structural extraction only.

It does NOT decide whether content is educationally relevant.
That responsibility belongs to the classifier.

Architecture
------------
PDF
 ↓
Loader
 ↓
Extractor
 ↓
RawDocument
 ↓
Classifier
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from app.core.exceptions import AppException
from app.core.logging import logger
from app.ingestion.raw_document import (
    BlockType,
    BoundingBox,
    RawBlock,
    RawDocument,
    RawDocumentMetadata,
    RawPage,
)


class PDFExtractor:
    """
    Extract structural content from PDF documents using Docling.

    The extractor converts Docling's internal representation into
    application-owned RawDocument models so that the rest of the
    ingestion pipeline remains independent of Docling.
    """

    def __init__(
        self,
        *,
        do_ocr: bool = False,
        do_table_structure: bool = True,
    ) -> None:
        """
        Initialize the PDF extractor.

        Parameters
        ----------
        do_ocr:
            Whether OCR should be enabled.

            Disabled by default because ICAI's digital PDFs generally
            contain an actual text layer and OCR is significantly more
            expensive in CPU and memory usage.

        do_table_structure:
            Whether Docling should perform table-structure analysis.
        """

        pipeline_options = PdfPipelineOptions()

        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def extract(
        self,
        file_path: Path,
        *,
        page_range: tuple[int, int] | None = None,
    ) -> RawDocument:
        """
        Extract a PDF into a RawDocument.

        Parameters
        ----------
        file_path:
            Path to the PDF document.

        page_range:
            Optional inclusive page range.

            Example:
                (1, 3)

            extracts pages 1 through 3.

        Returns
        -------
        RawDocument
            Application-owned structural representation of the PDF.

        Raises
        ------
        AppException
            If extraction fails.
        """

        logger.info(
            f"Starting PDF extraction: {file_path}"
        )

        try:
            if page_range is None:
                result = self.converter.convert(
                    file_path,
                )
            else:
                result = self.converter.convert(
                    file_path,
                    page_range=page_range,
                )

            document = result.document

            raw_document = self._build_raw_document(
                document=document,
                file_path=file_path,
            )

            logger.info(
                f"PDF extraction completed successfully: "
                f"{file_path}"
            )

            return raw_document

        except Exception as exc:
            logger.exception(
                f"PDF extraction failed: {file_path}"
            )

            raise AppException(
                message=f"PDF extraction failed: {file_path}",
                status_code=422,
            ) from exc

    def _build_raw_document(
        self,
        *,
        document: object,
        file_path: Path,
    ) -> RawDocument:
        """
        Convert a Docling document into RawDocument.

        This method builds the page structure first and then places
        extracted blocks onto their corresponding pages.

        Docling provenance is used to determine the physical PDF page
        containing each extracted item.
        """

        pages: dict[int, RawPage] = {}

        docling_pages = getattr(
            document,
            "pages",
            {},
        )

        for page_number, page in docling_pages.items():
            page_number_int = int(page_number)

            page_size = getattr(
                page,
                "size",
                None,
            )

            if page_size is None:
                logger.warning(
                    f"Page {page_number_int} has no size information."
                )
                continue

            width = float(
                getattr(page_size, "width")
            )

            height = float(
                getattr(page_size, "height")
            )

            pages[page_number_int] = RawPage(
                page_number=page_number_int,
                width=width,
                height=height,
            )

        for item, _level in document.iterate_items():
            provenance = getattr(
                item,
                "prov",
                None,
            )

            if not provenance:
                logger.warning(
                    "Skipping item without provenance: "
                    f"{getattr(item, 'self_ref', '<unknown>')}"
                )
                continue

            page_number = provenance[0].page_no

            if page_number not in pages:
                logger.warning(
                    f"Skipping item with unknown page: "
                    f"{page_number}"
                )
                continue

            page_height = pages[page_number].height

            block = self._convert_item(
                item=item,
                document=document,
                page_height=page_height,
            )

            if block is None:
                continue

            pages[block.page].blocks.append(block)

        sorted_pages = [
            pages[page_number]
            for page_number in sorted(pages)
        ]

        parser_version = (
            importlib.metadata.version("docling")
        )

        metadata = RawDocumentMetadata(
            filename=file_path.name,
            title=file_path.stem,
            pages=len(sorted_pages),
            parser="docling",
            parser_version=parser_version,
        )

        return RawDocument(
            metadata=metadata,
            pages=sorted_pages,
        )

    def _convert_item(
        self,
        *,
        item: object,
        document: object,
        page_height: float,
    ) -> RawBlock | None:
        """
        Convert one Docling item into a RawBlock.

        Items without usable provenance are ignored by the caller.

        The method intentionally keeps the mapping conservative:
        Docling-specific semantics are translated into the application's
        generic structural BlockType values.
        """

        provenance = getattr(
            item,
            "prov",
            None,
        )

        if not provenance:
            return None

        page_number = int(
            provenance[0].page_no
        )

        bbox = getattr(
            provenance[0],
            "bbox",
            None,
        )

        normalized_bbox: BoundingBox | None = None

        if bbox is not None:
            normalized_bbox = self._convert_bbox(
                bbox,
                page_height=page_height,
            )

        label = str(
            getattr(
                item,
                "label",
                "",
            )
        ).lower()

        block_type = self._map_block_type(
            label
        )

        text = self._extract_text(
            item=item,
            document=document,
        )

        block_id = str(
            getattr(
                item,
                "self_ref",
                f"page-{page_number}-unknown",
            )
        )

        metadata: dict[str, object] = {
            "docling_label": label,
        }

        return RawBlock(
            id=block_id,
            page=page_number,
            block_type=block_type,
            text=text,
            bbox=normalized_bbox,
            metadata=metadata,
        )

    @staticmethod
    def _map_block_type(
        label: str,
    ) -> BlockType:
        """
        Map a Docling item label to the application's BlockType.
        """

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

    @staticmethod
    def _extract_text(
        *,
        item: object,
        document: object,
    ) -> str:
        """
        Extract human-readable text from a Docling item.

        Tables are exported as Markdown because Markdown preserves
        enough table structure for downstream classification and
        chunking.

        Pictures use their caption when available.
        """

        label = str(
            getattr(
                item,
                "label",
                "",
            )
        ).lower()

        if label == "table":
            try:
                return str(
                    item.export_to_markdown(
                        document
                    )
                )
            except Exception:
                logger.warning(
                    "Failed to export table as Markdown."
                )
                return ""

        if label in {
            "picture",
            "chart",
        }:
            try:
                return str(
                    item.caption_text(
                        document
                    )
                )
            except Exception:
                return ""

        text = getattr(
            item,
            "text",
            None,
        )

        if text is not None:
            return str(text)

        return ""

    @staticmethod
    def _convert_bbox(
        bbox: object,
        *,
        page_height: float,
    ) -> BoundingBox:
        """
        Convert Docling's bounding box into the application's
        normalized top-left coordinate system.

        Docling may represent coordinates using a bottom-left origin.

        Example:

            PDF coordinate system

            (0, page_height)
                  ┌──────────────┐
                  │              │
                  │    BLOCK     │
                  │              │
                  └──────────────┘
            (0,0)

        Our application uses:

            (0,0)
                  ┌──────────────┐
                  │    BLOCK     │
                  │              │
                  └──────────────┘
                         (page_width, page_height)

        Therefore, for bottom-left coordinates:

            top    = page_height - original_top
            bottom = page_height - original_bottom
        """

        coord_origin = str(
            getattr(
                bbox,
                "coord_origin",
                "",
            )
        ).lower()

        left = float(
            bbox.l
        )

        right = float(
            bbox.r
        )

        top = float(
            bbox.t
        )

        bottom = float(
            bbox.b
        )

        if "bottomleft" in coord_origin:
            top = page_height - top
            bottom = page_height - bottom

        return BoundingBox(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        )