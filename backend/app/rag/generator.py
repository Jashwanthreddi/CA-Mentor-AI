"""
Gemini generation layer for CA Mentor AI.

Flow:

    Reranked Evidence
            ↓
       GeminiGenerator
            ↓
    Grounded Answer
            ↓
      Atomic Claims
            ↓
    Evidence References

The generator is responsible ONLY for answer generation.

It does NOT:
    - retrieve documents
    - rerank documents
    - verify claims
    - decide factual correctness

Those responsibilities belong to other pipeline stages.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.rag.generation_prompt import build_generation_prompt
from app.rag.reranker import RerankedChunk


# ============================================================
# OUTPUT SCHEMAS
# ============================================================


class AtomicClaim(BaseModel):
    """
    A single factual statement extracted from the generated answer.

    Claims are intentionally separated because a later NLI verifier
    will independently check each claim against the retrieved evidence.
    """

    claim: str = Field(
        description="One atomic factual statement made by the answer."
    )

    evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Chunk IDs from the provided evidence that directly support "
            "this claim."
        ),
    )


class GenerationOutput(BaseModel):
    """
    Structured output requested from Gemini.
    """

    answer: str = Field(
        description=(
            "A concise educational answer grounded strictly in the "
            "provided evidence."
        )
    )

    atomic_claims: list[AtomicClaim] = Field(
        default_factory=list,
        description=(
            "Atomic factual claims extracted from the answer, with "
            "supporting evidence chunk IDs."
        )
    )


class EvidenceReference(BaseModel):
    """
    Evidence metadata returned to the application layer.
    """

    chunk_id: str
    rank: int
    reranker_score: float
    page_start: int | None = None
    page_end: int | None = None


class GenerationResult(BaseModel):
    """
    Final generation result consumed by later pipeline stages.
    """

    question: str

    answer: str

    atomic_claims: list[AtomicClaim]

    evidence: list[EvidenceReference]

    model: str


# ============================================================
# GENERATOR
# ============================================================


class GeminiGenerator:
    """
    Generates grounded answers using Gemini.

    Input:
        User question
        Reranked chunks

    Output:
        GenerationResult

    The generator does not verify whether the generated claims
    are actually entailed by the evidence. That will be handled
    later by the NLI/self-grading stage.
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    DEFAULT_MAX_OUTPUT_TOKENS = 1200

    DEFAULT_TEMPERATURE = 0.2

    def __init__(
        self,
        model_name: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        client: Any | None = None,
    ) -> None:

        if max_output_tokens < 1:
            raise ValueError(
                "max_output_tokens must be at least 1."
            )

        if temperature < 0:
            raise ValueError(
                "temperature must be >= 0."
            )

        self.model_name = (
            model_name
            or getattr(settings, "GEMINI_MODEL", "")
            or self.DEFAULT_MODEL
        )

        if not self.model_name.strip():
            raise ValueError(
                "Gemini model name cannot be empty."
            )

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        self._client = client

    # --------------------------------------------------------
    # CLIENT
    # --------------------------------------------------------

    @property
    def client(self) -> Any:

        if self._client is None:

            if not settings.GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY is not configured. "
                    "Add it to the project root .env file."
                )

            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai is not installed. "
                    "Install it with: "
                    "python -m pip install -U google-genai"
                ) from exc

            logger.info(
                "Initializing Gemini client."
            )

            self._client = genai.Client(
                api_key=settings.GOOGLE_API_KEY
            )

        return self._client

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    def generate(
        self,
        question: str,
        candidates: list[RerankedChunk],
    ) -> GenerationResult:

        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        if not candidates:
            raise ValueError(
                "At least one reranked evidence chunk is required."
            )

        logger.info(
            "Gemini generation started. "
            f"question={question!r}, "
            f"evidence_count={len(candidates)}"
        )

        prompt = build_generation_prompt(
            question=question,
            candidates=candidates,
        )

        response = self._generate_content(prompt)

        parsed = self._parse_response(response)

        evidence = self._build_evidence_references(
            candidates
        )

        result = GenerationResult(
            question=question,
            answer=parsed.answer.strip(),
            atomic_claims=parsed.atomic_claims,
            evidence=evidence,
            model=self.model_name,
        )

        self._validate_claim_references(
            result.atomic_claims,
            candidates,
        )

        logger.info(
            "Gemini generation completed. "
            f"claims={len(result.atomic_claims)}, "
            f"evidence={len(result.evidence)}"
        )

        return result

    # --------------------------------------------------------
    # GEMINI REQUEST
    # --------------------------------------------------------

    def _generate_content(
        self,
        prompt: str,
    ) -> Any:

        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. "
                "Install it with: "
                "python -m pip install -U google-genai"
            ) from exc

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GenerationOutput,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        try:

            return self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

        except Exception as exc:

            logger.exception(
                "Gemini generation failed."
            )

            raise RuntimeError(
                f"Gemini generation failed: {exc}"
            ) from exc

    # --------------------------------------------------------
    # RESPONSE PARSING
    # --------------------------------------------------------

    @staticmethod
    def _parse_response(
        response: Any,
    ) -> GenerationOutput:

        parsed = getattr(
            response,
            "parsed",
            None,
        )

        if parsed is not None:

            if isinstance(
                parsed,
                GenerationOutput,
            ):
                return parsed

            try:
                return GenerationOutput.model_validate(
                    parsed
                )
            except Exception:
                pass

        response_text = getattr(
            response,
            "text",
            None,
        )

        if not response_text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        try:

            return GenerationOutput.model_validate_json(
                response_text
            )

        except Exception as exc:

            raise RuntimeError(
                "Unable to parse Gemini structured output."
            ) from exc

    # --------------------------------------------------------
    # EVIDENCE REFERENCES
    # --------------------------------------------------------

    @staticmethod
    def _build_evidence_references(
        candidates: list[RerankedChunk],
    ) -> list[EvidenceReference]:

        references: list[EvidenceReference] = []

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):

            chunk = candidate.chunk.chunk

            references.append(
                EvidenceReference(
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    reranker_score=float(
                        candidate.reranker_score
                    ),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )

        return references

    # --------------------------------------------------------
    # CLAIM VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def _validate_claim_references(
        claims: list[AtomicClaim],
        candidates: list[RerankedChunk],
    ) -> None:

        available_ids = {
            candidate.chunk.chunk.chunk_id
            for candidate in candidates
        }

        for claim in claims:

            invalid_ids = set(
                claim.evidence_chunk_ids
            ) - available_ids

            if invalid_ids:

                raise RuntimeError(
                    "Gemini returned claim references "
                    "to chunks that were not supplied "
                    f"as evidence: {sorted(invalid_ids)}"
                )