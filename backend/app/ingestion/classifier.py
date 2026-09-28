"""
Educational Classifier

Purpose
-------
Convert structurally extracted RawDocument blocks into
EducationalElement objects.

The classifier is intentionally deterministic in V1.

Responsibilities
----------------
- Identify educational content types.
- Track meaningful educational hierarchy.
- Propagate learning-outcome context to list items.
- Detect examples / illustrations.
- Ignore repeated running headers and footers.
- Preserve source block provenance.

The classifier does NOT:
- group multiple blocks into knowledge units,
- perform semantic chunking,
- call an LLM,
- create embeddings,
- upload anything.

Those responsibilities belong to later pipeline stages.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass

from app.core.logging import logger
from app.ingestion.models import EducationalElement, EducationalType
from app.ingestion.raw_document import BlockType, RawBlock, RawDocument


# ---------------------------------------------------------------------------
# Classification patterns
# ---------------------------------------------------------------------------

DEFINITION_PATTERNS = (
    r"\bis defined as\b",
    r"\bmeans\b",
    r"\brefers to\b",
    r"\bdefinition\b",
    r"\bcan be defined as\b",
    r"\bis known as\b",
)

PROCEDURE_PATTERNS = (
    r"\bsteps?\b",
    r"\bprocedure\b",
    r"\bprocedure for\b",
    r"\bmethod\b",
    r"\bhow to\b",
    r"\bpreparation of\b",
)

FORMULA_PATTERNS = (
    r"\bformula\b",
    r"\bformulae\b",
    r"\bequation\b",
)

RULE_PATTERNS = (
    r"\brule\b",
    r"\brules\b",
    r"\bshall\b",
    r"\bmust\b",
    r"\baccording to\b",
    r"\bas per\b",
)

EXCEPTION_PATTERNS = (
    r"\bexception\b",
    r"\bexceptions\b",
    r"\bexcept\b",
    r"\bunless\b",
    r"\bprovided that\b",
)

IMPORTANT_PATTERNS = (
    r"\bimportant\b",
    r"\bnote\b",
    r"\bkey point\b",
    r"\bimportant point\b",
    r"\bremember\b",
)

SUMMARY_PATTERNS = (
    r"\bsummary\b",
    r"\bsummaries\b",
    r"\brecap\b",
    r"\bkey takeaways\b",
    r"\bin brief\b",
)

LEARNING_OUTCOME_PATTERNS = (
    r"\blearning outcomes?\b",
    r"\blearning objectives?\b",
    r"\bobjectives?\b",
    r"\bafter studying\b",
    r"\bafter completion\b",
    r"\byou will be able to\b",
)

PRACTICE_QUESTION_PATTERNS = (
    r"\bpractice questions?\b",
    r"\bself[- ]assessment\b",
    r"\btest yourself\b",
    r"\bquestions for practice\b",
    r"\btrue or false\b",
    r"\bmultiple choice questions?\b",
    r"\bmcqs?\b",
)

EXAM_QUESTION_PATTERNS = (
    r"\bexam questions?\b",
    r"\bprevious examination\b",
    r"\bprevious year\b",
    r"\bquestion paper\b",
    r"\bexamination\b",
)

CASE_STUDY_PATTERNS = (
    r"\bcase study\b",
    r"\bcase studies\b",
    r"\bcase scenario\b",
)

MEMORY_TRICK_PATTERNS = (
    r"\bmnemoni[ck]\b",
    r"\bmemory trick\b",
    r"\bremember this\b",
    r"\bquick memory\b",
)

STRUCTURAL_HEADING_PATTERN = re.compile(
    r"^\s*(?:theoretical framework)\s*$",
    re.IGNORECASE,
)
# ---------------------------------------------------------------------------
# Structural patterns
# ---------------------------------------------------------------------------

CHAPTER_LABEL_PATTERN = re.compile(
    r"^\s*chapter\s*$",
    re.IGNORECASE,
)

CHAPTER_NUMBER_PATTERN = re.compile(
    r"^\s*(\d+)\s*$",
)

UNIT_PATTERN = re.compile(
    r"^\s*unit\s*[-–—:]?\s*(\d+)?\s*(.*)$",
    re.IGNORECASE,
)

NUMBERED_SECTION_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)+)\s+(.+?)\s*$",
)

EXAMPLE_PATTERN = re.compile(
    r"^\s*(?:example|illustration)\s*[-:]?\s*\d*\s*$",
    re.IGNORECASE,
)

EXAMPLE_INLINE_PATTERN = re.compile(
    r"^\s*(?:example|illustration)\s+\d+[\s:.–—-]*",
    re.IGNORECASE,
)

LEARNING_OUTCOME_HEADING_PATTERN = re.compile(
    r"^\s*learning outcomes?\s*$",
    re.IGNORECASE,
)

LEARNING_OUTCOME_INTRO_PATTERN = re.compile(
    r"^\s*(?:after studying|after completion|you will be able to)\b",
    re.IGNORECASE,
)


@dataclass
class _HierarchyState:
    """
    Mutable hierarchy state used while walking the document.

    The hierarchy is intentionally represented as a list of strings
    because EducationalElement already exposes hierarchy_path as a
    simple ordered path.
    """

    chapter: str | None = None
    module: str | None = None
    unit: str | None = None
    topic: str | None = None
    subtopic: str | None = None

    def reset_below_chapter(self) -> None:
        self.module = None
        self.unit = None
        self.topic = None
        self.subtopic = None

    def reset_below_unit(self) -> None:
        self.topic = None
        self.subtopic = None

    def reset_below_topic(self) -> None:
        self.subtopic = None

    def path(self) -> list[str]:
        return [
            value
            for value in (
                self.chapter,
                self.module,
                self.unit,
                self.topic,
                self.subtopic,
            )
            if value
        ]


class EducationalClassifier:
    """
    Deterministically classify RawDocument blocks into educational elements.
    """

    def classify(self, document: RawDocument) -> list[EducationalElement]:
        logger.info(
            f"Classifying document: {document.metadata.filename}"
        )

        blocks = [
            block
            for page in document.pages
            for block in page.blocks
        ]

        repeated_headers = self._detect_repeated_headers(document)

        state = _HierarchyState()
        elements: list[EducationalElement] = []

        learning_outcome_context = False

        for block in blocks:
            text = self._clean_text(block.text)

            if not text:
                continue

            # ---------------------------------------------------------------
            # Running page furniture
            # ---------------------------------------------------------------

            if block.block_type in {
                BlockType.HEADER,
                BlockType.FOOTER,
            }:
                continue

            if self._is_repeated_header(block, text, repeated_headers):
                continue

            # ---------------------------------------------------------------
            # Structural chapter detection
            # ---------------------------------------------------------------

            if CHAPTER_LABEL_PATTERN.match(text):
                state.chapter = "Chapter"
                state.reset_below_chapter()
                learning_outcome_context = False
                continue

            chapter_number_match = CHAPTER_NUMBER_PATTERN.match(text)

            if (
                chapter_number_match
                and state.chapter == "Chapter"
            ):
                state.chapter = f"Chapter {chapter_number_match.group(1)}"
                continue


            if STRUCTURAL_HEADING_PATTERN.match(text):
                learning_outcome_context = False
                continue

            # ---------------------------------------------------------------
            # Unit detection
            # ---------------------------------------------------------------

            unit_match = UNIT_PATTERN.match(text)

            if unit_match and (
                text.lower().startswith("unit")
            ):
                unit_number = unit_match.group(1)
                unit_title = unit_match.group(2).strip()

                if unit_number and unit_title:
                    state.unit = (
                        f"Unit {unit_number} - {unit_title}"
                    )
                elif unit_title:
                    state.unit = unit_title
                else:
                    state.unit = text

                state.reset_below_unit()
                learning_outcome_context = False
                continue

            # ---------------------------------------------------------------
            # Numbered section / topic detection
            # ---------------------------------------------------------------

            numbered_match = NUMBERED_SECTION_PATTERN.match(text)

            if (
                numbered_match
                and block.block_type == BlockType.HEADING
            ):
                section_number = numbered_match.group(1)
                section_title = numbered_match.group(2).strip()

                depth = section_number.count(".")

                if depth == 1:
                    state.topic = f"{section_number} {section_title}"
                    state.subtopic = None

                elif depth >= 2:
                    state.subtopic = (
                        f"{section_number} {section_title}"
                    )

                learning_outcome_context = False
                continue

            # ---------------------------------------------------------------
            # Learning outcomes
            # ---------------------------------------------------------------

            if LEARNING_OUTCOME_HEADING_PATTERN.match(text):
                learning_outcome_context = True

                elements.append(
                    self._create_element(
                        block=block,
                        educational_type=EducationalType.LEARNING_OUTCOME,
                        hierarchy_path=state.path(),
                        confidence=0.98,
                    )
                )

                continue

            if LEARNING_OUTCOME_INTRO_PATTERN.match(text):
                learning_outcome_context = True

                elements.append(
                    self._create_element(
                        block=block,
                        educational_type=EducationalType.LEARNING_OUTCOME,
                        hierarchy_path=state.path(),
                        confidence=0.98,
                    )
                )

                continue

            # ---------------------------------------------------------------
            # Example / illustration detection
            # ---------------------------------------------------------------

            if (
                block.block_type == BlockType.HEADING
                and (
                    EXAMPLE_PATTERN.match(text)
                    or EXAMPLE_INLINE_PATTERN.match(text)
                )
            ):
                learning_outcome_context = False

                elements.append(
                    self._create_element(
                        block=block,
                        educational_type=EducationalType.ILLUSTRATION,
                        hierarchy_path=state.path(),
                        confidence=0.98,
                    )
                )

                continue

            # ---------------------------------------------------------------
            # List items inside learning-outcome context
            # ---------------------------------------------------------------

            if (
                learning_outcome_context
                and block.block_type == BlockType.LIST
            ):
                elements.append(
                    self._create_element(
                        block=block,
                        educational_type=EducationalType.LEARNING_OUTCOME,
                        hierarchy_path=state.path(),
                        confidence=0.95,
                    )
                )
                continue

            # A normal paragraph after the learning-outcome section
            # terminates that context.
            if (
                learning_outcome_context
                and block.block_type not in {
                    BlockType.LIST,
                    BlockType.CAPTION,
                }
            ):
                learning_outcome_context = False

            # ---------------------------------------------------------------
            # Regular educational classification
            # ---------------------------------------------------------------

            educational_type, confidence = self._classify_block(
                block,
                text,
            )

            elements.append(
                self._create_element(
                    block=block,
                    educational_type=educational_type,
                    hierarchy_path=state.path(),
                    confidence=confidence,
                )
            )

        logger.info(
            f"Classification completed: {len(elements)} elements"
        )

        return elements

    # -----------------------------------------------------------------------
    # Block classification
    # -----------------------------------------------------------------------

    def _classify_block(
        self,
        block: RawBlock,
        text: str,
    ) -> tuple[EducationalType, float]:

        if block.block_type == BlockType.TABLE:
            return EducationalType.TABLE, 0.99

        if block.block_type == BlockType.IMAGE:
            return EducationalType.DIAGRAM, 0.70

        if block.block_type == BlockType.FORMULA:
            return EducationalType.FORMULA, 0.99

        if block.block_type == BlockType.LIST:
            return EducationalType.CONCEPT, 0.65

        if block.block_type == BlockType.CAPTION:
            return EducationalType.IMPORTANT_POINT, 0.60

        # ---------------------------------------------------------------
        # High-confidence semantic patterns
        # ---------------------------------------------------------------

        if self._matches(text, DEFINITION_PATTERNS):
            return EducationalType.DEFINITION, 0.90

        if self._matches(text, PROCEDURE_PATTERNS):
            return EducationalType.PROCEDURE, 0.86

        if self._matches(text, FORMULA_PATTERNS):
            return EducationalType.FORMULA, 0.92

        if self._matches(text, RULE_PATTERNS):
            return EducationalType.RULE, 0.84

        if self._matches(text, EXCEPTION_PATTERNS):
            return EducationalType.EXCEPTION, 0.88

        if self._matches(text, IMPORTANT_PATTERNS):
            return EducationalType.IMPORTANT_POINT, 0.85

        if self._matches(text, SUMMARY_PATTERNS):
            return EducationalType.SUMMARY, 0.90

        if self._matches(text, PRACTICE_QUESTION_PATTERNS):
            return EducationalType.PRACTICE_QUESTION, 0.95

        if self._matches(text, EXAM_QUESTION_PATTERNS):
            return EducationalType.EXAM_QUESTION, 0.90

        if self._matches(text, CASE_STUDY_PATTERNS):
            return EducationalType.CASE_STUDY, 0.95

        if self._matches(text, MEMORY_TRICK_PATTERNS):
            return EducationalType.MEMORY_TRICK, 0.95

        # ---------------------------------------------------------------
        # Generic educational content
        # ---------------------------------------------------------------

        if block.block_type == BlockType.HEADING:
            return EducationalType.CONCEPT, 0.75

        return EducationalType.CONCEPT, 0.60

    # -----------------------------------------------------------------------
    # Repeated header detection
    # -----------------------------------------------------------------------

    def _detect_repeated_headers(
        self,
        document: RawDocument,
    ) -> set[str]:

        candidates: list[str] = []

        for page in document.pages:
            for block in page.blocks:
                if block.block_type != BlockType.HEADING:
                    continue

                text = self._clean_text(block.text)

                if not text:
                    continue

                # Only short headings are considered possible running
                # headers. This avoids accidentally removing repeated
                # educational sections.
                if len(text) > 80:
                    continue

                candidates.append(text.casefold())

        counts = Counter(candidates)

        # A heading appearing on multiple pages is suspicious,
        # but we intentionally require >= 2 occurrences.
        #
        # Parser-level context can make the final decision later.
        return {
            text
            for text, count in counts.items()
            if count >= 2
        }

    def _is_repeated_header(
        self,
        block: RawBlock,
        text: str,
        repeated_headers: set[str],
    ) -> bool:

        if block.block_type != BlockType.HEADING:
            return False

        normalized = text.casefold()

        if normalized not in repeated_headers:
            return False

        # Structural headings such as Unit 1, 1.1, etc. must never
        # be discarded merely because they repeat.
        if UNIT_PATTERN.match(text):
            return False

        if NUMBERED_SECTION_PATTERN.match(text):
            return False

        if CHAPTER_LABEL_PATTERN.match(text):
            return False

        if LEARNING_OUTCOME_HEADING_PATTERN.match(text):
            return False

        if EXAMPLE_PATTERN.match(text):
            return False

        # Short repeated non-numbered headings are treated as likely
        # running headers.
        return True

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _matches(
        text: str,
        patterns: tuple[str, ...],
    ) -> bool:

        return any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in patterns
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _create_element(
        block: RawBlock,
        educational_type: EducationalType,
        hierarchy_path: list[str],
        confidence: float,
    ) -> EducationalElement:

        return EducationalElement(
            id=f"eku-{uuid.uuid4().hex}",
            educational_type=educational_type,
            content=" ".join(block.text.split()),
            page_start=block.page,
            page_end=block.page,
            hierarchy_path=list(hierarchy_path),
            source_block_ids=[block.id],
            confidence=confidence,
        )


classifier = EducationalClassifier()