"""
Prompt construction for CA Mentor AI generation.

The prompt deliberately forces Gemini to stay inside the
retrieved evidence.

The model is NOT allowed to use outside knowledge.
"""

from __future__ import annotations

from app.rag.reranker import RerankedChunk


SYSTEM_INSTRUCTIONS = """
You are the educational answer-generation component of CA Mentor AI.

Your task is to answer the student's question using ONLY the supplied
evidence passages.

STRICT GROUNDING RULES:

1. Use only information present in the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent facts, definitions, examples, dates, numbers, or formulas.
4. If the evidence does not sufficiently answer the question, say that
   the available study material does not contain enough information
   to answer confidently.
5. Keep the answer educational, clear, and concise.
6. Do not mention the retrieval system, vector database, embeddings,
   reranker, or internal implementation.
7. Every factual statement in the answer must be represented as an
   atomic claim.
8. Every atomic claim must reference one or more supplied evidence
   chunk IDs.
9. Never create or modify evidence chunk IDs.
10. Do not cite evidence that does not support the claim.

ANSWER STYLE:

- Start with a direct answer.
- Use short paragraphs or bullets when useful.
- Prefer simple explanations suitable for a CA student.
- Preserve important accounting terminology from the source material.
- Do not over-explain when the evidence is simple.

IMPORTANT:

You are generating a draft answer.

You are NOT responsible for deciding whether the claims are factually
entailed. A separate verification component will independently check
the claims against the evidence.
"""


def build_generation_prompt(
    question: str,
    candidates: list[RerankedChunk],
) -> str:

    if not question.strip():
        raise ValueError(
            "question cannot be empty."
        )

    if not candidates:
        raise ValueError(
            "At least one evidence candidate is required."
        )

    evidence_sections: list[str] = []

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):

        chunk = candidate.chunk.chunk

        evidence_sections.append(
            f"""
--- EVIDENCE {rank} ---

chunk_id: {chunk.chunk_id}

reranker_score: {candidate.reranker_score:.6f}

page_start: {chunk.page_start}

page_end: {chunk.page_end}

educational_type: {chunk.educational_type}

content:
{chunk.text}
"""
        )

    evidence_text = "\n".join(
        evidence_sections
    )

    return f"""
{SYSTEM_INSTRUCTIONS}

STUDENT QUESTION:

{question}

SUPPLIED EVIDENCE:

{evidence_text}

TASK:

Answer the student's question using only the supplied evidence.

Return the required structured JSON response with:

- answer
- atomic_claims

For every atomic claim, include the exact chunk_id values from the
supplied evidence that support that claim.
"""