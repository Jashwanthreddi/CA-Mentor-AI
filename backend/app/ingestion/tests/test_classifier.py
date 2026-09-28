from app.ingestion.classifier import EducationalClassifier
from app.ingestion.models import EducationalType
from app.ingestion.raw_document import (
    BlockType,
    BoundingBox,
    RawBlock,
    RawDocument,
    RawDocumentMetadata,
    RawPage,
)


def make_document(blocks: list[RawBlock]) -> RawDocument:
    return RawDocument(
        metadata=RawDocumentMetadata(
            filename="test.pdf",
            title="Test Document",
            pages=1,
            language="English",
            parser="test",
            parser_version="1.0",
        ),
        pages=[
            RawPage(
                page_number=1,
                width=1000,
                height=1400,
                blocks=blocks,
            )
        ],
    )


def make_block(
    block_id: str,
    text: str,
    block_type: BlockType = BlockType.PARAGRAPH,
    page: int = 1,
) -> RawBlock:
    return RawBlock(
        id=block_id,
        page=page,
        block_type=block_type,
        text=text,
        bbox=BoundingBox(
            left=10,
            top=10,
            right=500,
            bottom=100,
        ),
    )


def test_definition_classification():
    document = make_document(
        [
            make_block(
                "b1",
                "Accounting is defined as the systematic recording "
                "of financial transactions."
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 1
    assert elements[0].educational_type == EducationalType.DEFINITION


def test_accounting_process_is_not_procedure():
    """
    Regression test.

    The word 'process' alone must not cause a paragraph to become
    a procedure.
    """

    document = make_document(
        [
            make_block(
                "b1",
                "Accounting is the process of identifying transactions."
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 1
    assert elements[0].educational_type == EducationalType.CONCEPT


def test_procedure_classification():
    document = make_document(
        [
            make_block(
                "b1",
                "The following are the steps for preparing the statement."
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert elements[0].educational_type == EducationalType.PROCEDURE


def test_formula_classification():
    document = make_document(
        [
            make_block(
                "b1",
                "The formula for calculating profit is revenue minus expenses."
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert elements[0].educational_type == EducationalType.FORMULA


def test_table_classification():
    document = make_document(
        [
            make_block(
                "b1",
                "Revenue | Expenses",
                BlockType.TABLE,
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert elements[0].educational_type == EducationalType.TABLE


def test_formula_block_classification():
    document = make_document(
        [
            make_block(
                "b1",
                "Assets = Liabilities + Capital",
                BlockType.FORMULA,
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert elements[0].educational_type == EducationalType.FORMULA


def test_learning_outcome_heading():
    document = make_document(
        [
            make_block(
                "b1",
                "LEARNING OUTCOMES",
                BlockType.HEADING,
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 1
    assert elements[0].educational_type == EducationalType.LEARNING_OUTCOME


def test_learning_outcome_intro():
    document = make_document(
        [
            make_block(
                "b1",
                "After studying this unit, you would be able to:",
                BlockType.HEADING,
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 1
    assert elements[0].educational_type == EducationalType.LEARNING_OUTCOME


def test_learning_outcome_list_items():
    document = make_document(
        [
            make_block(
                "b1",
                "LEARNING OUTCOMES",
                BlockType.HEADING,
            ),
            make_block(
                "b2",
                "After studying this unit, you would be able to:",
                BlockType.HEADING,
            ),
            make_block(
                "b3",
                "Understand meaning and significance of accounting.",
                BlockType.LIST,
            ),
            make_block(
                "b4",
                "Understand the distinction between bookkeeping and accounting.",
                BlockType.LIST,
            ),
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 4

    assert all(
        element.educational_type == EducationalType.LEARNING_OUTCOME
        for element in elements
    )


def test_learning_outcomes_do_not_enter_hierarchy():
    document = make_document(
        [
            make_block(
                "b1",
                "UNIT - 1 MEANING AND SCOPE OF ACCOUNTING",
                BlockType.HEADING,
            ),
            make_block(
                "b2",
                "LEARNING OUTCOMES",
                BlockType.HEADING,
            ),
            make_block(
                "b3",
                "After studying this unit, you would be able to:",
                BlockType.HEADING,
            ),
            make_block(
                "b4",
                "Understand meaning of accounting.",
                BlockType.LIST,
            ),
            make_block(
                "b5",
                "1.1 INTRODUCTION",
                BlockType.HEADING,
            ),
        ]
    )

    elements = EducationalClassifier().classify(document)

    # Unit heading is structural and therefore not emitted.
    # Learning outcome heading + intro + bullet are emitted.
    # 1.1 INTRODUCTION is structural and therefore not emitted.
    assert len(elements) == 3

    assert all(
        element.educational_type == EducationalType.LEARNING_OUTCOME
        for element in elements
    )


def test_example_heading_is_illustration():
    document = make_document(
        [
            make_block(
                "b1",
                "Example 1",
                BlockType.HEADING,
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 1
    assert elements[0].educational_type == EducationalType.ILLUSTRATION


def test_illustration_heading_is_illustration():
    document = make_document(
        [
            make_block(
                "b1",
                "Illustration 1",
                BlockType.HEADING,
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 1
    assert elements[0].educational_type == EducationalType.ILLUSTRATION


def test_chapter_structure_is_not_emitted_as_concepts():
    document = make_document(
        [
            make_block(
                "b1",
                "CHAPTER",
                BlockType.HEADING,
            ),
            make_block(
                "b2",
                "1",
                BlockType.HEADING,
            ),
            make_block(
                "b3",
                "THEORETICAL FRAMEWORK",
                BlockType.HEADING,
            ),
        ]
    )

    elements = EducationalClassifier().classify(document)

    # CHAPTER, 1 and THEORETICAL FRAMEWORK are structural context.
    assert len(elements) == 0


def test_unit_is_not_emitted_as_concept():
    document = make_document(
        [
            make_block(
                "b1",
                "CHAPTER",
                BlockType.HEADING,
            ),
            make_block(
                "b2",
                "1",
                BlockType.HEADING,
            ),
            make_block(
                "b3",
                "UNIT - 1 MEANING AND SCOPE OF ACCOUNTING",
                BlockType.HEADING,
            ),
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 0


def test_hierarchy_chapter_unit_topic():
    document = make_document(
        [
            make_block(
                "b1",
                "CHAPTER",
                BlockType.HEADING,
            ),
            make_block(
                "b2",
                "1",
                BlockType.HEADING,
            ),
            make_block(
                "b3",
                "UNIT - 1 MEANING AND SCOPE OF ACCOUNTING",
                BlockType.HEADING,
            ),
            make_block(
                "b4",
                "1.1 INTRODUCTION",
                BlockType.HEADING,
            ),
            make_block(
                "b5",
                "Accounting is concerned with economic activities.",
                BlockType.PARAGRAPH,
            ),
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 1

    element = elements[0]

    assert element.educational_type == EducationalType.CONCEPT

    assert element.hierarchy_path == [
        "Chapter 1",
        "Unit 1 - MEANING AND SCOPE OF ACCOUNTING",
        "1.1 INTRODUCTION",
    ]


def test_repeated_running_header_is_ignored():
    document = RawDocument(
        metadata=RawDocumentMetadata(
            filename="test.pdf",
            title="Test Document",
            pages=2,
            language="English",
            parser="test",
            parser_version="1.0",
        ),
        pages=[
            RawPage(
                page_number=1,
                width=1000,
                height=1400,
                blocks=[
                    make_block(
                        "b1",
                        "THEORETICAL FRAMEWORK",
                        BlockType.HEADING,
                        page=1,
                    ),
                    make_block(
                        "b2",
                        "Accounting is important.",
                        BlockType.PARAGRAPH,
                        page=1,
                    ),
                ],
            ),
            RawPage(
                page_number=2,
                width=1000,
                height=1400,
                blocks=[
                    make_block(
                        "b3",
                        "THEORETICAL FRAMEWORK",
                        BlockType.HEADING,
                        page=2,
                    ),
                    make_block(
                        "b4",
                        "Accounting records economic transactions.",
                        BlockType.PARAGRAPH,
                        page=2,
                    ),
                ],
            ),
        ],
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 2

    assert all(
        element.content != "THEORETICAL FRAMEWORK"
        for element in elements
    )


def test_headers_and_footers_are_ignored():
    document = make_document(
        [
            make_block(
                "b1",
                "Running Header",
                BlockType.HEADER,
            ),
            make_block(
                "b2",
                "Useful accounting concept.",
                BlockType.PARAGRAPH,
            ),
            make_block(
                "b3",
                "Page 1",
                BlockType.FOOTER,
            ),
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert len(elements) == 1
    assert elements[0].content == "Useful accounting concept."


def test_source_block_provenance_is_preserved():
    document = make_document(
        [
            make_block(
                "source-123",
                "Accounting is defined as a systematic process.",
            )
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert elements[0].source_block_ids == ["source-123"]


def test_empty_blocks_are_ignored():
    document = make_document(
        [
            make_block(
                "b1",
                "",
            ),
            make_block(
                "b2",
                "   ",
            ),
        ]
    )

    elements = EducationalClassifier().classify(document)

    assert elements == []