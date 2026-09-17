from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, Field

from ..reranker import RerankerResult

logger = logging.getLogger(__name__)


class EvidenceVerdict(BaseModel):
    """Structured decision returned by the Evidence Gate."""
    allowed: bool
    confidence: float
    reason: str
    supporting_chunks: List[RerankerResult] = Field(default_factory=list)


class EvidenceGate:
    """
    Evidence Gate ensures that the generation model only responds when
    sufficient, reliable, relevant evidence exists in the retrieved context.
    Prevents answering false premises, hallucinations, or ungrounded queries.
    """

    def __init__(
        self,
        min_supporting_chunks: int = 1,
        min_confidence: float = 0.40,
        min_rerank_threshold: float = -4.0,
    ) -> None:
        self.min_supporting_chunks = min_supporting_chunks
        self.min_confidence = min_confidence
        self.min_rerank_threshold = min_rerank_threshold

    def evaluate(
        self,
        query: str,
        candidates: Sequence[RerankerResult],
    ) -> EvidenceVerdict:
        q_clean = query.strip()

        if not candidates:
            return EvidenceVerdict(
                allowed=False,
                confidence=0.0,
                reason="No candidate chunks were retrieved for this query.",
                supporting_chunks=[],
            )

        # Check that top candidates cross the minimum relevance threshold
        viable_chunks = [
            c for c in candidates if c.rerank_score >= self.min_rerank_threshold
        ]

        if len(viable_chunks) < self.min_supporting_chunks:
            return EvidenceVerdict(
                allowed=False,
                confidence=0.15,
                reason=(
                    "Retrieved documents scored below the relevance threshold; "
                    "insufficient evidence to construct a grounded response."
                ),
                supporting_chunks=[],
            )

        # Premise verification: check keyword presence to guard against false premises
        q_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", q_clean.lower()))
        # Remove common query stop words
        common_stops = {
            "what", "when", "where", "which", "who", "whom", "whose", "why",
            "how", "does", "explain", "tell", "about", "describe", "compare",
            "with", "from", "that", "this", "these", "those", "have", "were",
        }
        key_q_words = q_words - common_stops

        if not key_q_words:
            key_q_words = q_words

        combined_text = " ".join(c.text.lower() for c in viable_chunks)
        overlap_words = {w for w in key_q_words if w in combined_text}

        if key_q_words:
            overlap_ratio = len(overlap_words) / len(key_q_words)
        else:
            overlap_ratio = 1.0

        # If less than 20% of query keywords appear anywhere in retrieved context
        if overlap_ratio < 0.20 and len(key_q_words) >= 3:
            return EvidenceVerdict(
                allowed=False,
                confidence=0.20,
                reason=(
                    "The retrieved context lacks key terminology and premise evidence "
                    "for the query. Refusing to hallucinate."
                ),
                supporting_chunks=[],
            )

        # Compute confidence score
        best_score = max(c.rerank_score for c in viable_chunks)
        # Normalize heuristic confidence between 0.0 and 1.0
        confidence = min(1.0, max(0.4, 0.5 + (best_score / 10.0) + (overlap_ratio * 0.3)))

        if confidence < self.min_confidence:
            return EvidenceVerdict(
                allowed=False,
                confidence=confidence,
                reason=f"Evidence confidence {confidence:.2f} is below minimum threshold {self.min_confidence:.2f}.",
                supporting_chunks=[],
            )

        return EvidenceVerdict(
            allowed=True,
            confidence=confidence,
            reason="Sufficient grounded evidence found in retrieved chunks.",
            supporting_chunks=viable_chunks,
        )
