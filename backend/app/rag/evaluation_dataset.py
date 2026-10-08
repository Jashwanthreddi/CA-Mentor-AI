from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    question: str
    relevant_chunk_ids: tuple[str, ...]


EVALUATION_DATASET: tuple[RetrievalEvaluationCase, ...] = (

    # ------------------------------------------------------------------
    # 1. ACCOUNTING DEFINITION
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What is accounting?",
        relevant_chunk_ids=(
            "chunk-f6f33e1b-111b-5284-8563-aa5d50ca3541",
            "chunk-de1639ec-7c26-57ad-82c7-5bba7c6aa5c0",
        ),
    ),

    # ------------------------------------------------------------------
    # 2. AICPA DEFINITION
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What definition of accounting was formulated by the American Institute of Certified Public Accountants?",
        relevant_chunk_ids=(
            "chunk-f6f33e1b-111b-5284-8563-aa5d50ca3541",
        ),
    ),

    # ------------------------------------------------------------------
    # 3. FUNCTIONS OF ACCOUNTING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What are the main functions of accounting?",
        relevant_chunk_ids=(
            "chunk-daaff07d-e0d4-5b81-9cb3-80effe4ad61e",
        ),
    ),

    # ------------------------------------------------------------------
    # 4. OBJECTIVES OF ACCOUNTING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What are the objectives of accounting?",
        relevant_chunk_ids=(
            "chunk-c89f1314-fba4-505e-a07c-0d96364b35a3",
        ),
    ),

    # ------------------------------------------------------------------
    # 5. BOOK-KEEPING DEFINITION
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What is book-keeping?",
        relevant_chunk_ids=(
            "chunk-b9f9a905-d493-50ac-93f4-4639344f719f",
        ),
    ),

    # ------------------------------------------------------------------
    # 6. BOOK-KEEPING VS ACCOUNTING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What is the difference between book-keeping and accounting?",
        relevant_chunk_ids=(
            "chunk-5bab1a87-5b69-5f0c-a01f-7fb6d6bba631",
        ),
    ),

    # ------------------------------------------------------------------
    # 7. SUB-FIELDS OF ACCOUNTING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What are the various sub-fields of accounting?",
        relevant_chunk_ids=(
            "chunk-a3f3f20a-7816-5992-8d6b-4b96f751ff73",
        ),
    ),

    # ------------------------------------------------------------------
    # 8. FINANCIAL ACCOUNTING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What is financial accounting?",
        relevant_chunk_ids=(
            "chunk-a3f3f20a-7816-5992-8d6b-4b96f751ff73",
        ),
    ),

    # ------------------------------------------------------------------
    # 9. USERS OF ACCOUNTING INFORMATION
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="Who are the users of accounting information?",
        relevant_chunk_ids=(
            "chunk-3d118fae-f0a3-52a6-89c7-3e8c25d37eab",
            "chunk-ac5a114f-c6d5-5d38-adf4-4c8e4d7a7c73",
        ),
    ),

    # ------------------------------------------------------------------
    # 10. INTERNAL AND EXTERNAL USERS
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What are the two categories of users of accounting information?",
        relevant_chunk_ids=(
            "chunk-3d118fae-f0a3-52a6-89c7-3e8c25d37eab",
        ),
    ),

    # ------------------------------------------------------------------
    # 11. MANAGEMENT AS A USER
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="How does management use accounting information?",
        relevant_chunk_ids=(
            "chunk-5ac1e0cf-26de-5d95-8954-f9369221940c",
            "chunk-3d118fae-f0a3-52a6-89c7-3e8c25d37eab",
        ),
    ),

    # ------------------------------------------------------------------
    # 12. RECORDING FUNCTION
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What is the recording function of accounting?",
        relevant_chunk_ids=(
            "chunk-2362bf4f-1292-53c5-87f8-6f3480f6b8ed",
        ),
    ),

    # ------------------------------------------------------------------
    # 13. COMMUNICATION FUNCTION
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What is the communication function of accounting?",
        relevant_chunk_ids=(
            "chunk-bd0669ad-3658-594e-b4c7-30355a9fa9be",
        ),
    ),

    # ------------------------------------------------------------------
    # 14. ACCOUNTING PROCEDURE
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What are the two broad parts of the accounting procedure?",
        relevant_chunk_ids=(
            "chunk-2362bf4f-1292-53c5-87f8-6f3480f6b8ed",
            "chunk-de1639ec-7c26-57ad-82c7-5bba7c6aa5c0",
        ),
    ),

    # ------------------------------------------------------------------
    # 15. LIMITATIONS OF ACCOUNTING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What are the limitations of accounting?",
        relevant_chunk_ids=(
            "chunk-86302b26-02c3-5733-9f20-1075a173d6d9",
        ),
    ),

    # ------------------------------------------------------------------
    # 16. EVOLUTION OF ACCOUNTING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="How did accounting evolve historically?",
        relevant_chunk_ids=(
            "chunk-8d1d47e5-ae8d-55fe-8486-2eaa5e8e3360",
            "chunk-dcecc3df-7e21-5d85-9a63-adb69dcf48cf",
        ),
    ),

    # ------------------------------------------------------------------
    # 17. STEWARDSHIP ACCOUNTING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What is stewardship accounting?",
        relevant_chunk_ids=(
            "chunk-dcecc3df-7e21-5d85-9a63-adb69dcf48cf",
        ),
    ),

    # ------------------------------------------------------------------
    # 18. RELATED DISCIPLINES
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="Which disciplines are closely related to accounting?",
        relevant_chunk_ids=(
            "chunk-c11fdc80-79ef-5abc-959e-35d5bcde4184",
        ),
    ),

    # ------------------------------------------------------------------
    # 19. ACCOUNTING AND ECONOMICS
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What is the relationship between accounting and economics?",
        relevant_chunk_ids=(
            "chunk-c11fdc80-79ef-5abc-959e-35d5bcde4184",
        ),
    ),

    # ------------------------------------------------------------------
    # 20. ACCOUNTANT SERVICES
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What services do accountants provide to society?",
        relevant_chunk_ids=(
            "chunk-ee93192c-9a6e-552d-aacd-5cd9e8dc8a28",
            "chunk-8dee8367-ab17-5900-b565-d185a276c6c2",
            "chunk-39d3e565-3802-5b82-80b2-14d454c47f86",
        ),
    ),

    # ------------------------------------------------------------------
    # 21. TAXATION ROLE
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What role does an accountant play in taxation?",
        relevant_chunk_ids=(
            "chunk-8dee8367-ab17-5900-b565-d185a276c6c2",
        ),
    ),

    # ------------------------------------------------------------------
    # 22. ACCOUNTANT'S ROLE IN SOCIETY
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What is the role of an accountant in society?",
        relevant_chunk_ids=(
            "chunk-2775c9df-b95d-5f24-b476-a99079a3e4c6",
            "chunk-62f6012f-5a54-5f19-8089-81a3dc08db49",
        ),
    ),

    # ------------------------------------------------------------------
    # 23. ACCOUNTING AND DECISION-MAKING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="How does accounting help in decision-making?",
        relevant_chunk_ids=(
            "chunk-daaff07d-e0d4-5b81-9cb3-80effe4ad61e",
        ),
    ),

    # ------------------------------------------------------------------
    # 24. COMPLETE RECORDING IN BOOK-KEEPING
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="What does complete recording of transactions mean in book-keeping?",
        relevant_chunk_ids=(
            "chunk-d0b7428e-4b59-5a94-a189-c02948d89047",
        ),
    ),

    # ------------------------------------------------------------------
    # 25. WORKED EXAMPLE
    # ------------------------------------------------------------------
    RetrievalEvaluationCase(
        question="In the accounting example, how much did the individual invest and what transactions were performed?",
        relevant_chunk_ids=(
            "chunk-bafe79f8-8956-57eb-9ef1-d8e815158f0d",
        ),
    ),
)