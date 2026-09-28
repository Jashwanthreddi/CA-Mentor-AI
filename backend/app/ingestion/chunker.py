"""
Semantic Chunker for the CA Mentor AI ingestion pipeline.

Pipeline position:

EducationalKnowledgeUnit[]
        ↓
     Chunker
        ↓
DocumentChunk[]

The chunker operates on educational knowledge units rather than
raw PDF text.

Design principles
-----------------
1. Never chunk across EKU boundaries.
2. Preserve educational meaning.
3. Prefer paragraph boundaries.
4. Fall back to sentence boundaries.
5. Fall back to word boundaries when necessary.
6. Keep tables intact whenever possible.
7. Keep worked examples together whenever possible.
8. Keep individual questions independent.
9. Add controlled overlap only when an EKU is split.
10. Preserve EKU provenance.
11. Use deterministic chunk IDs.
12. Do not depend on an embedding provider.
"""

from __future__ import annotations

import re
from uuid import uuid4

from app.ingestion.models import (
    EducationalKnowledgeUnit,
    EducationalType,
    DocumentChunk,
)


class SemanticChunker:
    """
    Convert EducationalKnowledgeUnit objects into retrieval-ready
    DocumentChunk objects.

    V1 uses an approximate token estimator so the chunker remains
    independent of the future embedding model.
    """

    DEFAULT_TARGET_TOKENS = 700
    DEFAULT_MAX_TOKENS = 900
    DEFAULT_OVERLAP_TOKENS = 100
    DEFAULT_MIN_TOKENS = 100

    # Content that should normally remain as one semantic unit.
    ATOMIC_TYPES = {
        EducationalType.LEARNING_OUTCOME,
        EducationalType.PRACTICE_QUESTION,
        EducationalType.EXAM_QUESTION,
        EducationalType.FORMULA,
        EducationalType.MEMORY_TRICK,
    }

    def __init__(
        self,
        *,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        min_tokens: int = DEFAULT_MIN_TOKENS,
    ) -> None:
        """
        Initialize the chunker.

        Parameters
        ----------
        target_tokens:
            Preferred approximate size of a chunk.

        max_tokens:
            Maximum approximate size of a chunk.

        overlap_tokens:
            Approximate overlap between chunks when an EKU is split.

        min_tokens:
            Minimum useful chunk size before attempting to merge
            with surrounding content.
        """

        if target_tokens <= 0:
            raise ValueError(
                "target_tokens must be greater than zero."
            )

        if max_tokens < target_tokens:
            raise ValueError(
                "max_tokens must be greater than or equal to "
                "target_tokens."
            )

        if overlap_tokens < 0:
            raise ValueError(
                "overlap_tokens cannot be negative."
            )

        if overlap_tokens >= target_tokens:
            raise ValueError(
                "overlap_tokens must be smaller than target_tokens."
            )

        if min_tokens <= 0:
            raise ValueError(
                "min_tokens must be greater than zero."
            )

        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    # ========================================================
    # PUBLIC API
    # ========================================================

    def chunk(
        self,
        knowledge_units: list[EducationalKnowledgeUnit],
    ) -> list[DocumentChunk]:
        """
        Convert EKUs into retrieval-ready document chunks.

        Each EKU is processed independently.

        Returns
        -------
        list[DocumentChunk]
        """

        if not knowledge_units:
            return []

        chunks: list[DocumentChunk] = []

        for eku in knowledge_units:
            eku_chunks = self._chunk_eku(eku)

            chunks.extend(eku_chunks)

        return chunks

    # ========================================================
    # EKU CHUNKING
    # ========================================================

    def _chunk_eku(
        self,
        eku: EducationalKnowledgeUnit,
    ) -> list[DocumentChunk]:
        """
        Chunk one EKU without crossing its semantic boundary.
        """

        text = self._clean_text(eku.content)

        if not text:
            return []

        estimated_tokens = self.estimate_tokens(text)

        # Small EKUs remain completely intact.
        if estimated_tokens <= self.max_tokens:
            return [
                self._create_chunk(
                    eku=eku,
                    text=text,
                    chunk_index=0,
                    total_chunks=1,
                )
            ]

        # Atomic educational content should preferably remain intact.
        if eku.educational_type in self.ATOMIC_TYPES:
            return [
                self._create_chunk(
                    eku=eku,
                    text=text,
                    chunk_index=0,
                    total_chunks=1,
                )
            ]

        sections = self._split_semantically(text)

        if not sections:
            sections = [text]

        chunk_texts = self._pack_sections(sections)

        chunk_texts = self._apply_overlap(chunk_texts)

        # Safety check: overlap must never push a chunk beyond max.
        chunk_texts = self._enforce_max_size(chunk_texts)

        return [
            self._create_chunk(
                eku=eku,
                text=chunk_text,
                chunk_index=index,
                total_chunks=len(chunk_texts),
            )
            for index, chunk_text in enumerate(chunk_texts)
        ]

    # ========================================================
    # SEMANTIC SPLITTING
    # ========================================================

    def _split_semantically(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text using the following priority:

            paragraph
                ↓
            sentence
                ↓
            word
        """

        paragraphs = self._split_paragraphs(text)

        sections: list[str] = []

        for paragraph in paragraphs:
            if self.estimate_tokens(paragraph) <= self.max_tokens:
                sections.append(paragraph)
                continue

            sentences = self._split_sentences(paragraph)

            for sentence in sentences:
                if self.estimate_tokens(sentence) <= self.max_tokens:
                    sections.append(sentence)
                    continue

                # Extremely long sentence fallback.
                words = self._split_words(sentence)

                sections.extend(words)

        return sections

    def _split_paragraphs(
        self,
        text: str,
    ) -> list[str]:
        """Split text at paragraph boundaries."""

        paragraphs = re.split(
            r"\n\s*\n+",
            text,
        )

        return [
            self._clean_text(paragraph)
            for paragraph in paragraphs
            if self._clean_text(paragraph)
        ]

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text at sentence boundaries.

        This is deliberately conservative. It avoids trying to
        implement a full NLP sentence tokenizer in the ingestion
        layer.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9₹(])",
            text,
        )

        return [
            self._clean_text(sentence)
            for sentence in sentences
            if self._clean_text(sentence)
        ]

    def _split_words(
        self,
        text: str,
    ) -> list[str]:
        """
        Last-resort word-based splitting.

        Used only when a single sentence exceeds max_tokens.
        """

        words = text.split()

        if not words:
            return []

        pieces: list[str] = []
        current: list[str] = []

        for word in words:
            candidate = " ".join(
                current + [word]
            )

            if (
                current
                and self.estimate_tokens(candidate)
                > self.max_tokens
            ):
                pieces.append(
                    " ".join(current)
                )
                current = [word]
            else:
                current.append(word)

        if current:
            pieces.append(
                " ".join(current)
            )

        return pieces

    # ========================================================
    # PACKING
    # ========================================================

    def _pack_sections(
        self,
        sections: list[str],
    ) -> list[str]:
        """
        Pack semantic sections into chunks.

        Sections are added until the target size is reached.
        A section is never added if doing so would exceed the
        maximum size.
        """

        chunks: list[str] = []

        current_sections: list[str] = []
        current_tokens = 0

        for section in sections:
            section_tokens = self.estimate_tokens(
                section
            )

            # ------------------------------------------------
            # A single section larger than max is already
            # handled by the semantic splitting stage.
            # ------------------------------------------------

            if section_tokens > self.max_tokens:
                if current_sections:
                    chunks.append(
                        "\n\n".join(current_sections)
                    )

                    current_sections = []
                    current_tokens = 0

                chunks.append(section)

                continue

            # ------------------------------------------------
            # Empty buffer.
            # ------------------------------------------------

            if not current_sections:
                current_sections = [section]
                current_tokens = section_tokens
                continue

            proposed_tokens = (
                current_tokens + section_tokens
            )

            # ------------------------------------------------
            # Prefer target boundary.
            # ------------------------------------------------

            if proposed_tokens <= self.target_tokens:
                current_sections.append(section)
                current_tokens = proposed_tokens
                continue

            # ------------------------------------------------
            # We crossed target but still fit under max.
            #
            # If current chunk is too small, keep adding.
            # Otherwise close it.
            # ------------------------------------------------

            if (
                current_tokens < self.min_tokens
                and proposed_tokens <= self.max_tokens
            ):
                current_sections.append(section)
                current_tokens = proposed_tokens
                continue

            chunks.append(
                "\n\n".join(current_sections)
            )

            current_sections = [section]
            current_tokens = section_tokens

        if current_sections:
            chunks.append(
                "\n\n".join(current_sections)
            )

        return [
            self._clean_text(chunk)
            for chunk in chunks
            if self._clean_text(chunk)
        ]

    # ========================================================
    # OVERLAP
    # ========================================================

    def _apply_overlap(
        self,
        chunks: list[str],
    ) -> list[str]:
        """
        Add controlled overlap between neighboring chunks.

        Overlap is taken from the END of the previous chunk and
        prepended to the next chunk.

        If overlap cannot fit without exceeding max_tokens,
        it is reduced automatically.
        """

        if (
            len(chunks) <= 1
            or self.overlap_tokens == 0
        ):
            return chunks

        result: list[str] = [chunks[0]]

        for index in range(1, len(chunks)):
            previous = result[-1]
            current = chunks[index]

            overlap = self._tail_by_tokens(
                previous,
                self.overlap_tokens,
            )

            if not overlap:
                result.append(current)
                continue

            current_tokens = self.estimate_tokens(
                current
            )

            available_tokens = (
                self.max_tokens - current_tokens
            )

            if available_tokens <= 0:
                result.append(current)
                continue

            overlap = self._tail_by_tokens(
                previous,
                min(
                    self.overlap_tokens,
                    available_tokens,
                ),
            )

            if overlap:
                merged = (
                    f"{overlap}\n\n{current}"
                )

                if (
                    self.estimate_tokens(merged)
                    <= self.max_tokens
                ):
                    result.append(merged)
                else:
                    result.append(current)
            else:
                result.append(current)

        return result

    def _tail_by_tokens(
        self,
        text: str,
        token_count: int,
    ) -> str:
        """
        Return approximately the last token_count tokens.
        """

        words = text.split()

        if not words or token_count <= 0:
            return ""

        # V1 approximation: one whitespace-separated word
        # is approximately one token.
        return " ".join(
            words[-token_count:]
        )

    # ========================================================
    # MAXIMUM SIZE ENFORCEMENT
    # ========================================================

    def _enforce_max_size(
        self,
        chunks: list[str],
    ) -> list[str]:
        """
        Final safety pass.

        If overlap or another operation produces a chunk larger
        than max_tokens, split it again.
        """

        result: list[str] = []

        for chunk in chunks:
            if (
                self.estimate_tokens(chunk)
                <= self.max_tokens
            ):
                result.append(chunk)
                continue

            sections = self._split_semantically(
                chunk
            )

            packed = self._pack_sections(
                sections
            )

            result.extend(packed)

        return [
            self._clean_text(chunk)
            for chunk in result
            if self._clean_text(chunk)
        ]

    # ========================================================
    # CHUNK CREATION
    # ========================================================

    def _create_chunk(
        self,
        *,
        eku: EducationalKnowledgeUnit,
        text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> DocumentChunk:
        """
        Create a DocumentChunk while preserving EKU metadata.
        """

        cleaned_text = self._clean_text(text)

        return DocumentChunk(
            chunk_id=self._generate_chunk_id(),
            eku_id=eku.id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            text=cleaned_text,
            token_count=self.estimate_tokens(
                cleaned_text
            ),
            metadata=eku.metadata,
        )

    # ========================================================
    # TOKEN ESTIMATION
    # ========================================================

    @staticmethod
    def estimate_tokens(
        text: str,
    ) -> int:
        """
        Estimate token count for V1.

        This is intentionally provider-independent.

        Approximation:
            tokens ≈ words * 1.3

        The exact tokenizer will be introduced after the
        embedding model is finalized.
        """

        words = text.split()

        if not words:
            return 0

        return max(
            1,
            round(len(words) * 1.3),
        )

    # ========================================================
    # TEXT HELPERS
    # ========================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """Normalize whitespace without changing meaning."""

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

    # ========================================================
    # ID GENERATION
    # ========================================================

    @staticmethod
    def _generate_chunk_id() -> str:
        """Generate a globally unique chunk ID."""

        return f"chunk-{uuid4()}"