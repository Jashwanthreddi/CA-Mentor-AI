"""
Tests for the Educational Knowledge Unit parser.

The tests verify that the parser:

1. Handles empty input.
2. Creates EKUs from normal educational elements.
3. Keeps learning outcomes independent.
4. Groups normal explanatory content.
5. Groups illustrations with their supporting content.
6. Attaches tables to the current knowledge unit.
7. Creates standalone EKUs for standalone tables.
8. Preserves hierarchy.
9. Preserves source element IDs.
10. Preserves page ranges.
11. Creates unique EKU IDs.
12. Preserves educational metadata.
"""

from app.ingestion.models import (
    EducationalElement,
    EducationalType,
)
from app.ingestion.parser import EducationalParser


# ============================================================
# TEST HELPERS
# ============================================================


def make_element(
    *,
    element_id: str,
    educational_type: EducationalType,
    content: str,
    page_start: int = 1,
    page_end: int | None = None,
    hierarchy_path: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> EducationalElement:
    """Create a test EducationalElement."""

    return EducationalElement(
        id=element_id,
        educational_type=educational_type,
        content=content,
        page_start=page_start,
        page_end=page_end or page_start,
        hierarchy_path=hierarchy_path or [
            "Chapter 1",
            "Unit 1 - TEST",
            "1.1 INTRODUCTION",
        ],
        source_block_ids=[f"block-{element_id}"],
        metadata=metadata or {},
    )


# ============================================================
# EMPTY INPUT
# ============================================================


def test_parse_empty_elements_returns_empty_list() -> None:
    """Empty input should produce no EKUs."""

    parser = EducationalParser()

    result = parser.parse([])

    assert result == []


# ============================================================
# SINGLE ELEMENT
# ============================================================


def test_single_concept_creates_one_eku() -> None:
    """One concept should create one EKU."""

    parser = EducationalParser()

    element = make_element(
        element_id="element-1",
        educational_type=EducationalType.CONCEPT,
        content="Accounting is the process of recording financial transactions.",
    )

    result = parser.parse([element])

    assert len(result) == 1

    eku = result[0]

    assert eku.educational_type == EducationalType.CONCEPT
    assert (
        eku.content
        == "Accounting is the process of recording financial transactions."
    )


# ============================================================
# LEARNING OUTCOMES
# ============================================================


def test_learning_outcomes_remain_independent() -> None:
    """
    Learning outcomes should not be merged into one large EKU.
    """

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="lo-1",
            educational_type=EducationalType.LEARNING_OUTCOME,
            content="Understand the meaning of accounting.",
        ),
        make_element(
            element_id="lo-2",
            educational_type=EducationalType.LEARNING_OUTCOME,
            content="Identify the objectives of accounting.",
        ),
        make_element(
            element_id="lo-3",
            educational_type=EducationalType.LEARNING_OUTCOME,
            content="Explain the scope of accounting.",
        ),
    ]

    result = parser.parse(elements)

    assert len(result) == 3

    assert all(
        eku.educational_type
        == EducationalType.LEARNING_OUTCOME
        for eku in result
    )

    assert result[0].content == (
        "Understand the meaning of accounting."
    )

    assert result[1].content == (
        "Identify the objectives of accounting."
    )

    assert result[2].content == (
        "Explain the scope of accounting."
    )


# ============================================================
# NORMAL GROUPING
# ============================================================


def test_consecutive_explanatory_elements_are_grouped() -> None:
    """
    Consecutive explanatory elements under the same hierarchy
    should form one EKU.
    """

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="concept-1",
            educational_type=EducationalType.CONCEPT,
            content="Accounting records financial transactions.",
        ),
        make_element(
            element_id="concept-2",
            educational_type=EducationalType.CONCEPT,
            content="These transactions are classified systematically.",
        ),
        make_element(
            element_id="concept-3",
            educational_type=EducationalType.CONCEPT,
            content="The information is useful for decision making.",
        ),
    ]

    result = parser.parse(elements)

    assert len(result) == 1

    eku = result[0]

    assert "Accounting records financial transactions." in eku.content
    assert "These transactions are classified systematically." in eku.content
    assert "The information is useful for decision making." in eku.content

    assert eku.source_element_ids == [
        "concept-1",
        "concept-2",
        "concept-3",
    ]


# ============================================================
# DIFFERENT HIERARCHY
# ============================================================


def test_different_hierarchy_creates_separate_ekus() -> None:
    """A hierarchy change should create an educational boundary."""

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="topic-1",
            educational_type=EducationalType.CONCEPT,
            content="Content belonging to topic one.",
            hierarchy_path=[
                "Chapter 1",
                "Unit 1",
                "1.1 Topic One",
            ],
        ),
        make_element(
            element_id="topic-2",
            educational_type=EducationalType.CONCEPT,
            content="Content belonging to topic two.",
            hierarchy_path=[
                "Chapter 1",
                "Unit 1",
                "1.2 Topic Two",
            ],
        ),
    ]

    result = parser.parse(elements)

    assert len(result) == 2

    assert result[0].hierarchy_path == [
        "Chapter 1",
        "Unit 1",
        "1.1 Topic One",
    ]

    assert result[1].hierarchy_path == [
        "Chapter 1",
        "Unit 1",
        "1.2 Topic Two",
    ]


# ============================================================
# ILLUSTRATION / WORKED EXAMPLE
# ============================================================


def test_illustration_groups_following_supporting_content() -> None:
    """
    Illustration + explanation should become one WORKED_EXAMPLE EKU.
    """

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="example-heading",
            educational_type=EducationalType.ILLUSTRATION,
            content="Example 1",
        ),
        make_element(
            element_id="example-content",
            educational_type=EducationalType.CONCEPT,
            content="A business purchases machinery for ₹5,00,000.",
        ),
        make_element(
            element_id="example-explanation",
            educational_type=EducationalType.CONCEPT,
            content="The machinery is recorded as an asset.",
        ),
    ]

    result = parser.parse(elements)

    assert len(result) == 1

    eku = result[0]

    assert eku.educational_type == (
        EducationalType.WORKED_EXAMPLE
    )

    assert eku.title == "Example 1"

    assert "A business purchases machinery" in eku.content
    assert "The machinery is recorded as an asset." in eku.content

    assert eku.source_element_ids == [
        "example-heading",
        "example-content",
        "example-explanation",
    ]

    assert eku.metadata.has_example is True


# ============================================================
# TABLE ATTACHMENT
# ============================================================


def test_table_attaches_to_current_knowledge_unit() -> None:
    """
    A table appearing after explanatory content should be attached
    to the current EKU instead of creating a separate EKU.
    """

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="concept-1",
            educational_type=EducationalType.CONCEPT,
            content="Accounting information can be classified into different types.",
        ),
        make_element(
            element_id="table-1",
            educational_type=EducationalType.TABLE,
            content="Financial Accounting | Management Accounting",
            metadata={
                "title": "Types of Accounting",
                "rows": "2",
                "columns": "2",
            },
        ),
        make_element(
            element_id="concept-2",
            educational_type=EducationalType.CONCEPT,
            content="Each type serves a different purpose.",
        ),
    ]

    result = parser.parse(elements)

    assert len(result) == 1

    eku = result[0]

    assert eku.educational_type == EducationalType.CONCEPT

    assert len(eku.tables) == 1

    table = eku.tables[0]

    assert table.title == "Types of Accounting"
    assert table.rows == 2
    assert table.columns == 2

    assert eku.metadata.has_table is True


# ============================================================
# STANDALONE TABLE
# ============================================================


def test_standalone_table_creates_table_eku() -> None:
    """A table with no current context should become its own EKU."""

    parser = EducationalParser()

    element = make_element(
        element_id="table-1",
        educational_type=EducationalType.TABLE,
        content="Assets | Liabilities | Equity",
        metadata={
            "title": "Accounting Elements",
            "rows": "3",
            "columns": "3",
        },
    )

    result = parser.parse([element])

    assert len(result) == 1

    eku = result[0]

    assert eku.educational_type == EducationalType.TABLE
    assert len(eku.tables) == 1

    assert eku.tables[0].title == "Accounting Elements"
    assert eku.tables[0].rows == 3
    assert eku.tables[0].columns == 3


# ============================================================
# DIAGRAM
# ============================================================


def test_standalone_diagram_creates_diagram_eku() -> None:
    """A standalone diagram should become a diagram EKU."""

    parser = EducationalParser()

    element = make_element(
        element_id="diagram-1",
        educational_type=EducationalType.DIAGRAM,
        content="Accounting cycle diagram",
        metadata={
            "caption": "The accounting cycle",
            "image_type": "diagram",
        },
    )

    result = parser.parse([element])

    assert len(result) == 1

    eku = result[0]

    assert eku.educational_type == EducationalType.DIAGRAM

    assert len(eku.images) == 1

    assert eku.images[0].caption == "The accounting cycle"
    assert eku.images[0].image_type == "diagram"

    assert eku.metadata.has_image is True


# ============================================================
# FORMULA
# ============================================================


def test_formula_is_preserved() -> None:
    """Formula content should remain intact."""

    parser = EducationalParser()

    element = make_element(
        element_id="formula-1",
        educational_type=EducationalType.FORMULA,
        content="Assets = Liabilities + Capital",
    )

    result = parser.parse([element])

    assert len(result) == 1

    eku = result[0]

    assert eku.educational_type == EducationalType.FORMULA

    assert eku.content == "Assets = Liabilities + Capital"

    assert eku.metadata.has_formula is True


# ============================================================
# DEFINITION
# ============================================================


def test_definition_is_preserved() -> None:
    """Definitions should remain identifiable."""

    parser = EducationalParser()

    element = make_element(
        element_id="definition-1",
        educational_type=EducationalType.DEFINITION,
        content="Accounting is defined as the systematic recording of transactions.",
    )

    result = parser.parse([element])

    assert len(result) == 1

    eku = result[0]

    assert eku.educational_type == EducationalType.DEFINITION
    assert "Accounting is defined" in eku.content


# ============================================================
# METADATA
# ============================================================


def test_metadata_is_preserved() -> None:
    """Classifier metadata should flow into the EKU metadata."""

    parser = EducationalParser()

    element = make_element(
        element_id="concept-1",
        educational_type=EducationalType.CONCEPT,
        content="Accounting concept.",
        metadata={
            "level": "foundation",
            "paper": "Paper 1",
            "module": "Module 1",
            "chapter": "Chapter 1",
            "unit": "Unit 1",
            "topic": "Introduction",
            "difficulty": "intermediate",
            "teaching_priority": "high",
            "embedding_strategy": "full",
        },
    )

    result = parser.parse([element])

    assert len(result) == 1

    metadata = result[0].metadata

    assert metadata.level.value == "foundation"
    assert metadata.paper == "Paper 1"
    assert metadata.module == "Module 1"
    assert metadata.chapter == "Chapter 1"
    assert metadata.unit == "Unit 1"
    assert metadata.topic == "Introduction"

    assert metadata.difficulty.value == "intermediate"
    assert metadata.teaching_priority.value == "high"
    assert metadata.embedding_strategy.value == "full"


# ============================================================
# PAGE RANGE
# ============================================================


def test_page_range_is_preserved() -> None:
    """EKU page range should cover all source elements."""

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="element-1",
            educational_type=EducationalType.CONCEPT,
            content="First part.",
            page_start=2,
            page_end=2,
        ),
        make_element(
            element_id="element-2",
            educational_type=EducationalType.CONCEPT,
            content="Second part.",
            page_start=3,
            page_end=3,
        ),
        make_element(
            element_id="element-3",
            educational_type=EducationalType.CONCEPT,
            content="Third part.",
            page_start=4,
            page_end=4,
        ),
    ]

    result = parser.parse(elements)

    assert len(result) == 1

    eku = result[0]

    assert eku.metadata.page_start == 2
    assert eku.metadata.page_end == 4


# ============================================================
# PROVENANCE
# ============================================================


def test_source_element_ids_are_preserved() -> None:
    """EKU must retain provenance back to classifier output."""

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="element-100",
            educational_type=EducationalType.CONCEPT,
            content="First content.",
        ),
        make_element(
            element_id="element-101",
            educational_type=EducationalType.CONCEPT,
            content="Second content.",
        ),
    ]

    result = parser.parse(elements)

    assert len(result) == 1

    assert result[0].source_element_ids == [
        "element-100",
        "element-101",
    ]


# ============================================================
# UNIQUE IDS
# ============================================================


def test_eku_ids_are_unique() -> None:
    """Every EKU should receive a unique identifier."""

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="lo-1",
            educational_type=EducationalType.LEARNING_OUTCOME,
            content="Understand accounting.",
        ),
        make_element(
            element_id="lo-2",
            educational_type=EducationalType.LEARNING_OUTCOME,
            content="Understand bookkeeping.",
        ),
        make_element(
            element_id="lo-3",
            educational_type=EducationalType.LEARNING_OUTCOME,
            content="Understand financial statements.",
        ),
    ]

    result = parser.parse(elements)

    ids = [eku.id for eku in result]

    assert len(ids) == 3
    assert len(set(ids)) == 3

    assert all(
        eku.id.startswith("eku-")
        for eku in result
    )


# ============================================================
# QUESTIONS REMAIN INDEPENDENT
# ============================================================


def test_practice_questions_remain_independent() -> None:
    """Practice questions should not be merged into concepts."""

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="concept-1",
            educational_type=EducationalType.CONCEPT,
            content="Accounting records business transactions.",
        ),
        make_element(
            element_id="question-1",
            educational_type=EducationalType.PRACTICE_QUESTION,
            content="What is the purpose of accounting?",
        ),
        make_element(
            element_id="question-2",
            educational_type=EducationalType.PRACTICE_QUESTION,
            content="State two objectives of accounting.",
        ),
    ]

    result = parser.parse(elements)

    assert len(result) == 3

    assert result[0].educational_type == (
        EducationalType.CONCEPT
    )

    assert result[1].educational_type == (
        EducationalType.PRACTICE_QUESTION
    )

    assert result[2].educational_type == (
        EducationalType.PRACTICE_QUESTION
    )


# ============================================================
# SUMMARY REMAINS INDEPENDENT
# ============================================================


def test_summary_remains_independent() -> None:
    """Summary content should become its own EKU."""

    parser = EducationalParser()

    elements = [
        make_element(
            element_id="concept-1",
            educational_type=EducationalType.CONCEPT,
            content="Accounting provides useful financial information.",
        ),
        make_element(
            element_id="summary-1",
            educational_type=EducationalType.SUMMARY,
            content="Accounting records and communicates financial information.",
        ),
    ]

    result = parser.parse(elements)

    assert len(result) == 2

    assert result[0].educational_type == (
        EducationalType.CONCEPT
    )

    assert result[1].educational_type == (
        EducationalType.SUMMARY
    )


# ============================================================
# TEXT CLEANING
# ============================================================


def test_parser_normalizes_excessive_whitespace() -> None:
    """Parser should clean unnecessary whitespace."""

    parser = EducationalParser()

    element = make_element(
        element_id="concept-1",
        educational_type=EducationalType.CONCEPT,
        content="  Accounting    is important.  ",
    )

    result = parser.parse([element])

    assert len(result) == 1

    assert result[0].content == (
        "Accounting is important."
    )


# ============================================================
# ICAI-STYLE MINI SCENARIO
# ============================================================


def test_icai_style_sequence() -> None:
    """
    Simulate the structure observed in the ICAI Foundation PDF:

        Learning Outcomes
        Introduction
        Example
        Example explanation
        Table
        Explanation
    """

    parser = EducationalParser()

    hierarchy = [
        "Chapter 1",
        "Unit 1 - MEANING AND SCOPE OF ACCOUNTING",
        "1.1 INTRODUCTION",
    ]

    elements = [
        make_element(
            element_id="lo-1",
            educational_type=EducationalType.LEARNING_OUTCOME,
            content="Understand the meaning of accounting.",
            hierarchy_path=[
                "Chapter 1",
                "Unit 1 - MEANING AND SCOPE OF ACCOUNTING",
            ],
        ),
        make_element(
            element_id="intro-1",
            educational_type=EducationalType.CONCEPT,
            content="Accounting is concerned with financial information.",
            hierarchy_path=hierarchy,
            page_start=2,
        ),
        make_element(
            element_id="example-1",
            educational_type=EducationalType.ILLUSTRATION,
            content="Example 1",
            hierarchy_path=hierarchy,
            page_start=3,
        ),
        make_element(
            element_id="example-content",
            educational_type=EducationalType.CONCEPT,
            content="A business records its transactions.",
            hierarchy_path=hierarchy,
            page_start=3,
        ),
        make_element(
            element_id="table-1",
            educational_type=EducationalType.TABLE,
            content="Assets | Liabilities",
            hierarchy_path=hierarchy,
            page_start=3,
            metadata={
                "title": "Accounting Elements",
                "rows": "2",
                "columns": "2",
            },
        ),
        make_element(
            element_id="example-explanation",
            educational_type=EducationalType.CONCEPT,
            content="The transaction affects the accounting equation.",
            hierarchy_path=hierarchy,
            page_start=3,
        ),
    ]

    result = parser.parse(elements)

    # Expected:
    #
    # EKU 1 -> learning outcome
    # EKU 2 -> introduction
    # EKU 3 -> worked example containing table

    assert len(result) == 3

    assert result[0].educational_type == (
        EducationalType.LEARNING_OUTCOME
    )

    assert result[1].educational_type == (
        EducationalType.CONCEPT
    )

    assert result[2].educational_type == (
        EducationalType.WORKED_EXAMPLE
    )

    assert result[2].metadata.has_example is True
    assert result[2].metadata.has_table is True

    assert result[2].tables[0].title == (
        "Accounting Elements"
    )

    assert result[2].source_element_ids == [
        "example-1",
        "example-content",
        "table-1",
        "example-explanation",
    ]