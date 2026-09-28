"""
Educational Knowledge Unit Parser.

Purpose
-------
Convert classified EducationalElement objects into meaningful
EducationalKnowledgeUnit objects.

Pipeline position:

EducationalElement[]
        ↓
     Parser
        ↓
EducationalKnowledgeUnit[]

Design principles
-----------------
1. Deterministic in V1.
2. No LLM dependency.
3. No database dependency.
4. No embedding dependency.
5. Preserve document hierarchy.
6. Preserve source provenance.
7. Group semantically related elements.
8. Keep learning outcomes independent.
9. Group illustrations with their supporting content.
10. Keep tables/images attached to the knowledge they support.
"""

from __future__ import annotations

import re
from uuid import uuid4

from app.ingestion.models import (
    ChunkMetadata,
    DifficultyLevel,
    EducationalElement,
    EducationalKnowledgeUnit,
    EducationalType,
    EmbeddingStrategy,
    ImageMetadata,
    TableMetadata,
    DocumentLevel,
    TeachingPriority,
)


class EducationalParser:
    """
    Convert classified educational elements into
    EducationalKnowledgeUnit objects.

    The parser is intentionally deterministic.

    Input:
        list[EducationalElement]

    Output:
        list[EducationalKnowledgeUnit]
    """

    PARSER_VERSION = "parser-v1"

    # --------------------------------------------------------
    # Types that should always remain independent
    # --------------------------------------------------------

    INDEPENDENT_TYPES = {
        EducationalType.LEARNING_OUTCOME,
        EducationalType.PRACTICE_QUESTION,
        EducationalType.EXAM_QUESTION,
        EducationalType.MEMORY_TRICK,
        EducationalType.SUMMARY,
    }

    # --------------------------------------------------------
    # Types that can be attached to the current knowledge unit
    # --------------------------------------------------------

    ATTACHABLE_TYPES = {
        EducationalType.TABLE,
        EducationalType.DIAGRAM,
        EducationalType.FLOWCHART,
    }

    # --------------------------------------------------------
    # Types considered explanatory content
    # --------------------------------------------------------

    EXPLANATORY_TYPES = {
        EducationalType.CONCEPT,
        EducationalType.DEFINITION,
        EducationalType.PROCEDURE,
        EducationalType.FORMULA,
        EducationalType.RULE,
        EducationalType.EXCEPTION,
        EducationalType.IMPORTANT_POINT,
        EducationalType.CASE_STUDY,
    }

    def __init__(
        self,
        *,
        default_level: str = "foundation",
        default_paper: str = "unknown",
    ) -> None:
        """
        Initialize the parser.

        default_level and default_paper are temporary fallbacks.

        The Metadata Builder will later enrich these values using
        the authoritative document metadata.
        """

        self.default_level = default_level
        self.default_paper = default_paper

    # ========================================================
    # PUBLIC API
    # ========================================================

    def parse(
        self,
        elements: list[EducationalElement],
    ) -> list[EducationalKnowledgeUnit]:
        """
        Convert classified educational elements into EKUs.

        Parameters
        ----------
        elements:
            Classified educational elements produced by classifier.py.

        Returns
        -------
        list[EducationalKnowledgeUnit]
            Semantically grouped educational knowledge units.
        """

        if not elements:
            return []

        knowledge_units: list[EducationalKnowledgeUnit] = []

        current_elements: list[EducationalElement] = []

        for element in elements:
            # ------------------------------------------------
            # Independent educational content
            # ------------------------------------------------

            if element.educational_type in self.INDEPENDENT_TYPES:
                self._flush_current(
                    current_elements=current_elements,
                    output=knowledge_units,
                )

                current_elements = []

                eku = self._create_eku(
                    [element]
                )

                knowledge_units.append(eku)

                continue

            # ------------------------------------------------
            # Illustration starts an example group
            # ------------------------------------------------

            if element.educational_type == EducationalType.ILLUSTRATION:
                self._flush_current(
                    current_elements=current_elements,
                    output=knowledge_units,
                )

                current_elements = [element]

                continue

            # ------------------------------------------------
            # Table / diagram / flowchart
            # ------------------------------------------------

            if element.educational_type in self.ATTACHABLE_TYPES:
                if current_elements:
                    current_elements.append(element)
                else:
                    # Standalone visual content.
                    eku = self._create_eku([element])
                    knowledge_units.append(eku)

                continue

            # ------------------------------------------------
            # Normal explanatory content
            # ------------------------------------------------

            if not current_elements:
                current_elements = [element]
                continue

            # ------------------------------------------------
            # Example mode
            # ------------------------------------------------

            if self._is_example_group(current_elements):
                if self._belongs_to_same_context(
                    current_elements[-1],
                    element,
                ):
                    current_elements.append(element)
                else:
                    self._flush_current(
                        current_elements=current_elements,
                        output=knowledge_units,
                    )

                    current_elements = [element]

                continue

            # ------------------------------------------------
            # Normal grouping
            # ------------------------------------------------

            if self._should_group(
                current_elements[-1],
                element,
            ):
                current_elements.append(element)
            else:
                self._flush_current(
                    current_elements=current_elements,
                    output=knowledge_units,
                )

                current_elements = [element]

        # ----------------------------------------------------
        # Flush remaining content
        # ----------------------------------------------------

        self._flush_current(
            current_elements=current_elements,
            output=knowledge_units,
        )

        return knowledge_units

    # ========================================================
    # GROUPING LOGIC
    # ========================================================

    def _should_group(
        self,
        previous: EducationalElement,
        current: EducationalElement,
    ) -> bool:
        """
        Determine whether two normal educational elements
        should belong to the same EKU.
        """

        # Different hierarchy usually means a semantic boundary.
        if previous.hierarchy_path != current.hierarchy_path:
            return False

        # Tables/visuals are handled separately.
        if current.educational_type in self.ATTACHABLE_TYPES:
            return True

        if previous.educational_type in self.ATTACHABLE_TYPES:
            return True

        # Same explanatory family can safely be grouped.
        if (
            previous.educational_type in self.EXPLANATORY_TYPES
            and current.educational_type in self.EXPLANATORY_TYPES
        ):
            return True

        # A concept following a formula/definition/etc. can
        # still belong to the same educational idea.
        if self._is_supporting_content(
            previous,
            current,
        ):
            return True

        return False

    def _belongs_to_same_context(
        self,
        previous: EducationalElement,
        current: EducationalElement,
    ) -> bool:
        """
        Determine whether content belongs to the current
        illustration/example group.
        """

        if previous.hierarchy_path != current.hierarchy_path:
            return False

        if current.educational_type in self.INDEPENDENT_TYPES:
            return False

        return True

    def _is_supporting_content(
        self,
        previous: EducationalElement,
        current: EducationalElement,
    ) -> bool:
        """
        Identify common supporting-content transitions.
        """

        valid_previous = {
            EducationalType.DEFINITION,
            EducationalType.FORMULA,
            EducationalType.RULE,
            EducationalType.EXCEPTION,
            EducationalType.IMPORTANT_POINT,
        }

        valid_current = {
            EducationalType.CONCEPT,
            EducationalType.DEFINITION,
            EducationalType.PROCEDURE,
            EducationalType.FORMULA,
            EducationalType.RULE,
            EducationalType.EXCEPTION,
            EducationalType.IMPORTANT_POINT,
        }

        return (
            previous.educational_type in valid_previous
            and current.educational_type in valid_current
        )

    def _is_example_group(
        self,
        elements: list[EducationalElement],
    ) -> bool:
        """
        Return True when the current buffer represents
        an illustration/example.
        """

        return any(
            element.educational_type
            == EducationalType.ILLUSTRATION
            for element in elements
        )

    # ========================================================
    # BUFFER MANAGEMENT
    # ========================================================

    def _flush_current(
        self,
        *,
        current_elements: list[EducationalElement],
        output: list[EducationalKnowledgeUnit],
    ) -> None:
        """
        Convert the current element buffer into an EKU.
        """

        if not current_elements:
            return

        eku = self._create_eku(current_elements)

        output.append(eku)

        current_elements.clear()

    # ========================================================
    # EKU CREATION
    # ========================================================

    def _create_eku(
        self,
        elements: list[EducationalElement],
    ) -> EducationalKnowledgeUnit:
        """
        Create one EKU from one or more educational elements.
        """

        if not elements:
            raise ValueError(
                "Cannot create an EKU from empty elements."
            )

        primary_type = self._resolve_educational_type(
            elements
        )

        content = self._combine_content(elements)

        title = self._build_title(
            elements,
            primary_type,
        )

        hierarchy_path = self._resolve_hierarchy(
            elements
        )

        source_element_ids = [
            element.id
            for element in elements
        ]

        page_start = min(
            element.page_start
            for element in elements
        )

        page_end = max(
            element.page_end
            for element in elements
        )

        tables = self._extract_table_metadata(
            elements
        )

        images = self._extract_image_metadata(
            elements
        )

        metadata = self._build_chunk_metadata(
            elements=elements,
            educational_type=primary_type,
            page_start=page_start,
            page_end=page_end,
            tables=tables,
            images=images,
        )

        return EducationalKnowledgeUnit(
            id=self._generate_eku_id(),
            title=title,
            educational_type=primary_type,
            content=content,
            metadata=metadata,
            hierarchy_path=hierarchy_path,
            source_element_ids=source_element_ids,
            tables=tables,
            images=images,
        )

    # ========================================================
    # TYPE RESOLUTION
    # ========================================================

    def _resolve_educational_type(
        self,
        elements: list[EducationalElement],
    ) -> EducationalType:
        """
        Determine the dominant educational type of a group.
        """

        types = [
            element.educational_type
            for element in elements
        ]

        # Illustration + explanatory content becomes
        # a worked example.
        if EducationalType.ILLUSTRATION in types:
            return EducationalType.WORKED_EXAMPLE

        # A standalone table remains a table.
        if all(
            element.educational_type
            == EducationalType.TABLE
            for element in elements
        ):
            return EducationalType.TABLE

        # A standalone diagram remains a diagram.
        if all(
            element.educational_type
            == EducationalType.DIAGRAM
            for element in elements
        ):
            return EducationalType.DIAGRAM

        # A standalone flowchart remains a flowchart.
        if all(
            element.educational_type
            == EducationalType.FLOWCHART
            for element in elements
        ):
            return EducationalType.FLOWCHART

        # If all elements have the same type, preserve it.
        if len(set(types)) == 1:
            return types[0]

        # Otherwise select the strongest semantic type.
        priority = [
            EducationalType.DEFINITION,
            EducationalType.PROCEDURE,
            EducationalType.FORMULA,
            EducationalType.RULE,
            EducationalType.EXCEPTION,
            EducationalType.IMPORTANT_POINT,
            EducationalType.CASE_STUDY,
            EducationalType.CONCEPT,
        ]

        for educational_type in priority:
            if educational_type in types:
                return educational_type

        return EducationalType.CONCEPT

    # ========================================================
    # CONTENT
    # ========================================================

    def _combine_content(
        self,
        elements: list[EducationalElement],
    ) -> str:
        """
        Combine element content while preserving reading order.
        """

        parts: list[str] = []

        for element in elements:
            text = self._clean_text(element.content)

            if not text:
                continue

            parts.append(text)

        return "\n\n".join(parts)

    # ========================================================
    # TITLE
    # ========================================================

    def _build_title(
        self,
        elements: list[EducationalElement],
        educational_type: EducationalType,
    ) -> str:
        """
        Build a compact title for the EKU.

        For illustrations, preserve the illustration heading.
        For other content, use the first meaningful sentence.
        """

        first_text = self._clean_text(
            elements[0].content
        )

        # Example:
        # "Example 1" / "Illustration 2"
        if educational_type == EducationalType.WORKED_EXAMPLE:
            return self._extract_example_title(
                elements
            )

        # Learning outcomes should preserve their content.
        if educational_type == EducationalType.LEARNING_OUTCOME:
            return self._shorten_title(first_text)

        # Tables often have useful first-line titles.
        if educational_type == EducationalType.TABLE:
            return self._shorten_title(first_text)

        # Generic title from first meaningful content.
        return self._shorten_title(first_text)

    def _extract_example_title(
        self,
        elements: list[EducationalElement],
    ) -> str:
        """
        Extract the example/illustration title.
        """

        for element in elements:
            text = self._clean_text(
                element.content
            )

            if re.match(
                r"^(example|illustration)\b",
                text,
                flags=re.IGNORECASE,
            ):
                return self._shorten_title(text)

        return "Worked Example"

    # ========================================================
    # HIERARCHY
    # ========================================================

    def _resolve_hierarchy(
        self,
        elements: list[EducationalElement],
    ) -> list[str]:
        """
        Preserve the hierarchy of the first meaningful element.

        The classifier is responsible for constructing hierarchy.
        The parser should not attempt to reconstruct it again.
        """

        for element in elements:
            if element.hierarchy_path:
                return list(element.hierarchy_path)

        return []

    # ========================================================
    # TABLE METADATA
    # ========================================================

    def _extract_table_metadata(
        self,
        elements: list[EducationalElement],
    ) -> list[TableMetadata]:
        """
        Extract table metadata from classifier-provided metadata.
        """

        tables: list[TableMetadata] = []

        for element in elements:
            if element.educational_type != EducationalType.TABLE:
                continue

            title = element.metadata.get("title")

            rows = self._safe_int(
                element.metadata.get("rows")
            )

            columns = self._safe_int(
                element.metadata.get("columns")
            )

            tables.append(
                TableMetadata(
                    title=title,
                    rows=rows,
                    columns=columns,
                )
            )

        return tables

    # ========================================================
    # IMAGE METADATA
    # ========================================================

    def _extract_image_metadata(
        self,
        elements: list[EducationalElement],
    ) -> list[ImageMetadata]:
        """
        Extract image/diagram metadata.
        """

        images: list[ImageMetadata] = []

        for element in elements:
            if element.educational_type not in {
                EducationalType.DIAGRAM,
                EducationalType.FLOWCHART,
            }:
                continue

            images.append(
                ImageMetadata(
                    caption=element.metadata.get(
                        "caption"
                    ),
                    image_type=element.metadata.get(
                        "image_type"
                    ),
                )
            )

        return images

    # ========================================================
    # CHUNK METADATA
    # ========================================================

    def _build_chunk_metadata(
        self,
        *,
        elements: list[EducationalElement],
        educational_type: EducationalType,
        page_start: int,
        page_end: int,
        tables: list[TableMetadata],
        images: list[ImageMetadata],
    ) -> ChunkMetadata:
        """
        Build metadata required by DocumentChunk.

        The Metadata Builder will later enrich/normalize
        authoritative document-level information.
        """

        first = elements[0]

        raw_metadata = first.metadata

        level = raw_metadata.get(
            "level",
            self.default_level,
        )

        paper = raw_metadata.get(
            "paper",
            self.default_paper,
        )

        module = raw_metadata.get("module")
        chapter = raw_metadata.get("chapter")
        unit = raw_metadata.get("unit")
        topic = raw_metadata.get("topic")
        subtopic = raw_metadata.get("subtopic")

        difficulty = self._parse_enum(
            DifficultyLevel,
            raw_metadata.get("difficulty"),
            DifficultyLevel.BEGINNER,
        )

        teaching_priority = self._parse_enum(
            TeachingPriority,
            raw_metadata.get("teaching_priority"),
            TeachingPriority.HIGH,
        )

        embedding_strategy = self._parse_enum(
            EmbeddingStrategy,
            raw_metadata.get("embedding_strategy"),
            EmbeddingStrategy.FULL,
        )

        has_formula = any(
            element.educational_type
            == EducationalType.FORMULA
            for element in elements
        )

        has_example = any(
            element.educational_type
            in {
                EducationalType.ILLUSTRATION,
                EducationalType.WORKED_EXAMPLE,
            }
            for element in elements
        )

        has_image = bool(images)

        has_table = bool(tables)

        return ChunkMetadata(
            level=self._parse_level(level),
            paper=paper,
            module=module,
            chapter=chapter,
            unit=unit,
            topic=topic,
            subtopic=subtopic,
            educational_type=educational_type,
            difficulty=difficulty,
            teaching_priority=teaching_priority,
            embedding_strategy=embedding_strategy,
            page_start=page_start,
            page_end=page_end,
            has_table=has_table,
            has_image=has_image,
            has_formula=has_formula,
            has_example=has_example,
            version=self.PARSER_VERSION,
        )

    # ========================================================
    # ENUM HELPERS
    # ========================================================

    @staticmethod
    def _parse_level(
        value: str,) -> DocumentLevel:
        """
        Safely convert a string to DocumentLevel.
        """

        try:
            return DocumentLevel(value.lower())
        except ValueError:
            return DocumentLevel.FOUNDATION

    def _parse_enum(
        self,
        enum_class,
        value: str | None,
        default,
    ):
        """
        Safely convert string values to Enum members.
        """

        if not value:
            return default

        try:
            return enum_class(value.lower())
        except ValueError:
            return default

    # ========================================================
    # TEXT HELPERS
    # ========================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """
        Normalize whitespace without changing meaning.
        """

        text = text.strip()

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text

    @staticmethod
    def _shorten_title(
        text: str,
        max_length: int = 120,
    ) -> str:
        """
        Create a compact title from content.
        """

        text = text.strip()

        if len(text) <= max_length:
            return text

        shortened = text[:max_length].rsplit(
            " ",
            1,
        )[0]

        return f"{shortened}..."

    @staticmethod
    def _safe_int(
        value: str | None,
    ) -> int:
        """
        Safely parse integer metadata.
        """

        if value is None:
            return 0

        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    # ========================================================
    # ID GENERATION
    # ========================================================

    @staticmethod
    def _generate_eku_id() -> str:
        """
        Generate a globally unique EKU identifier.
        """

        return f"eku-{uuid4()}"