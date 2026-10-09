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
        min_confidence: float = 0.20,
        min_rerank_threshold: float = -5.0,
    ) -> None:
        self.min_supporting_chunks = min_supporting_chunks
        self.min_confidence = min_confidence
        self.min_rerank_threshold = min_rerank_threshold

    def evaluate(
        self,
        query: str,
        candidates: Sequence[RerankerResult],
    ) -> EvidenceVerdict:
        import math
        q_clean = query.strip()

        if not candidates:
            return EvidenceVerdict(
                allowed=False,
                confidence=0.0,
                reason="No candidate chunks were retrieved for this query.",
                supporting_chunks=[],
            )

        # 1. Adversarial Injection and Jailbreak Guard
        ADVERSARIAL_PATTERNS = [
            r"ignore\s+(all\s+)?(previous|prior|system|factual|facts|evidence|context|rules)",
            r"system\s+(prompt|override|message|instruction)",
            r"you\s+are\s+now\s+dan",
            r"act\s+as\s+(an?\s+)?(unrestricted|jailbroken|dan)",
            r"forget\s+(that\s+)?you\s+are",
            r"declare\s+that\s+.*was\s+invented",
            r"bypass\s+authentication",
            r"reveal\s+(database\s+)?credentials",
            r"secret\s+administrator\s+passwords",
            r"manufacture\s+explosives",
            r"drop\s+table",
            r"select\s+\*\s+from",
            r"sql\s+injection",
            r"<script>",
            r"unrestricted\s+chatbot\s+without\s+citations",
            r"private\s+tenants\s+in\s+the\s+database",
            r"repeat\s+the\s+word\s+['\"]?access_granted",
        ]
        q_lower = q_clean.lower()
        for pat in ADVERSARIAL_PATTERNS:
            if re.search(pat, q_lower):
                return EvidenceVerdict(
                    allowed=False,
                    confidence=0.0,
                    reason="Adversarial instruction or security policy violation detected. Operating strictly as a grounded research assistant.",
                    supporting_chunks=[],
                )

        # Check that top candidates cross the minimum relevance threshold
        viable_chunks = [
            c for c in candidates
            if c.rerank_score >= self.min_rerank_threshold
            or (getattr(c, "keyword_score", None) is not None and c.keyword_score >= 3.0)
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

        # 2. Comprehensive stopword filtering for substantive entity verification
        STOPWORDS = {
            "what", "when", "where", "which", "who", "whom", "whose", "why",
            "how", "does", "explain", "tell", "about", "describe", "compare",
            "with", "from", "that", "this", "these", "those", "have", "were",
            "the", "for", "and", "are", "can", "you", "all", "not", "but",
            "any", "some", "each", "into", "over", "after", "before", "more",
            "also", "than", "been", "has", "had", "would", "could", "should",
            "its", "our", "their", "will", "out", "other", "give", "show",
            "provide", "find", "use", "used", "using", "between", "under", "per",
            "won", "score", "year", "time", "place", "make", "made", "good", "new"
        }
        q_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", q_lower)
        key_q_words = [w for w in q_tokens if w not in STOPWORDS]

        if not key_q_words:
            key_q_words = q_tokens

        combined_text = " ".join(c.text.lower() for c in viable_chunks)
        overlap_words = [w for w in key_q_words if w in combined_text]
        overlap_ratio = len(overlap_words) / max(1, len(key_q_words))

        # Check that at least one single chunk contains cohesive evidence (prevents cross-doc accidental word matching)
        max_chunk_overlap = max((len([w for w in key_q_words if w in c.text.lower()]) for c in viable_chunks), default=0)

        # 3. Evidence sufficiency check: unanswerable / out-of-domain detection
        if (
            len(overlap_words) == 0
            or (len(overlap_words) < 2 and len(key_q_words) >= 3)
            or overlap_ratio < 0.25
            or (len(key_q_words) >= 3 and max_chunk_overlap < 2 and overlap_ratio < 0.35)
        ):
            return EvidenceVerdict(
                allowed=False,
                confidence=round(overlap_ratio * 0.3, 2),
                reason=(
                    "The retrieved context lacks key terminology and premise evidence "
                    "for the query. Refusing to hallucinate."
                ),
                supporting_chunks=[],
            )

        # 4. Authentic calibrated confidence score (no artificial max(0.4, ...) floor)
        best_score = max((c.rerank_score for c in viable_chunks), default=0.0)
        if best_score > 1.0 or best_score < 0.0:
            sig_score = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, best_score))))
        else:
            sig_score = min(1.0, best_score * 20.0) if best_score < 0.05 else best_score

        confidence = round(min(1.0, (sig_score * 0.4) + (overlap_ratio * 0.6)), 2)

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
