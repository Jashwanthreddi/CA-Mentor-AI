"""
Generation evaluation dataset adapter.

The initial generation evaluation reuses the existing
retrieval evaluation questions.

This keeps Phase 1 retrieval ground truth unchanged while
allowing generation quality to be evaluated independently.
"""

from __future__ import annotations

from app.rag.evaluation_dataset import EVALUATION_DATASET


def get_generation_questions() -> list[str]:
    """
    Return questions used for generation evaluation.

    The existing evaluation dataset is intentionally reused
    so that retrieval and generation are evaluated on the
    same question set during this initial phase.
    """

    return [
        item.question
        for item in EVALUATION_DATASET
    ]