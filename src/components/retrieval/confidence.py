from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

STOPWORDS: Set[str] = {
    "a", "an", "the", "in", "on", "of", "for", "to", "at", "by", "with", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "if", "then", "else", "what", "which",
    "who", "whom", "this", "that", "these", "those", "how", "why", "where", "when",
    "can", "could", "will", "would", "should", "about", "into", "through", "during",
}


@dataclass
class RetrievalConfidence:
    """
    Candidate confidence assessment for retrieval escalation decisions.
    """
    score: float
    is_sufficient: bool
    term_coverage: float
    exact_identifier_matches: List[str]
    missing_identifiers: List[str]
    candidate_count: int
    top_score: float
    score_margin: float
    reason: str


class RetrievalConfidenceScorer:
    """
    Evaluates candidate confidence to determine whether the pipeline can
    answer directly or must escalate to a deeper retrieval tier.
    """

    def __init__(
        self,
        min_coverage_fast: float = 0.50,
        min_coverage_balanced: float = 0.35,
        min_top_score_bm25: float = 2.0,
    ) -> None:
        self.min_coverage_fast = min_coverage_fast
        self.min_coverage_balanced = min_coverage_balanced
        self.min_top_score_bm25 = min_top_score_bm25

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[Any],
        execution_mode: str = "fast",
        exact_identifiers: Optional[List[str]] = None,
    ) -> RetrievalConfidence:
        """
        Compute retrieval confidence from candidate texts, scores, and term coverage.
        """
        if not candidates:
            return RetrievalConfidence(
                score=0.0,
                is_sufficient=False,
                term_coverage=0.0,
                exact_identifier_matches=[],
                missing_identifiers=list(exact_identifiers or []),
                candidate_count=0,
                top_score=0.0,
                score_margin=0.0,
                reason="No candidates retrieved",
            )

        # Extract non-stopword query tokens (min length 2)
        q_tokens = [
            w for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query.lower())
            if w not in STOPWORDS
        ]

        top_candidates = candidates[:3]
        combined_text = " ".join(
            getattr(c, "text", "") or getattr(c, "snippet", "") or ""
            for c in top_candidates
        ).lower()

        # Measure term coverage
        hits = sum(1 for t in q_tokens if t in combined_text)
        term_cov = hits / max(1, len(q_tokens))

        # Check required exact identifiers
        req_ids = [i.lower() for i in (exact_identifiers or [])]
        matched_ids = [i for i in req_ids if i in combined_text]
        missing_ids = [i for i in req_ids if i not in combined_text]

        # Extract candidate score attributes
        scores: List[float] = []
        for c in top_candidates:
            sc = getattr(c, "score", 0.0)
            try:
                scores.append(float(sc))
            except Exception:
                scores.append(0.0)

        top_score = scores[0] if scores else 0.0
        score_margin = (scores[0] - scores[1]) if len(scores) > 1 else top_score

        # Confidence calculation
        coverage_weight = 0.50
        id_weight = 0.35
        score_weight = 0.15

        id_ratio = (len(matched_ids) / len(req_ids)) if req_ids else 1.0
        norm_score = min(1.0, top_score / 10.0) if execution_mode == "fast" else min(1.0, top_score)

        conf_score = (
            coverage_weight * term_cov
            + id_weight * id_ratio
            + score_weight * norm_score
        )

        # Sufficiency thresholding
        threshold = self.min_coverage_fast if execution_mode == "fast" else self.min_coverage_balanced
        is_sufficient = (
            term_cov >= threshold
            and (not req_ids or len(matched_ids) > 0)
            and (execution_mode != "fast" or top_score >= self.min_top_score_bm25 or term_cov >= 0.6)
        )

        reason = (
            f"Confidence {conf_score:.2f} (coverage: {term_cov:.2f}, "
            f"matched_ids: {len(matched_ids)}/{len(req_ids)}, top_score: {top_score:.2f})"
        )

        return RetrievalConfidence(
            score=conf_score,
            is_sufficient=is_sufficient,
            term_coverage=term_cov,
            exact_identifier_matches=matched_ids,
            missing_identifiers=missing_ids,
            candidate_count=len(candidates),
            top_score=top_score,
            score_margin=score_margin,
            reason=reason,
        )
