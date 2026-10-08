from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set

logger = logging.getLogger(__name__)

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
    Candidate confidence assessment for retrieval escalation and conditional reranking.
    """
    score: float
    tier: Literal["high", "medium", "low"] = "medium"
    is_sufficient: bool = True
    term_coverage: float = 0.0
    exact_identifier_matches: List[str] = field(default_factory=list)
    missing_identifiers: List[str] = field(default_factory=list)
    candidate_count: int = 0
    top_score: float = 0.0
    score_margin: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    reason: str = ""

    def __post_init__(self):
        if not self.reason and self.reasons:
            self.reason = "; ".join(self.reasons)


class RetrievalConfidenceScorer:
    """
    Evaluates candidate retrieval quality using measurable signals:
    - Term coverage (lexical overlap of non-stopword tokens)
    - Exact technical identifier presence
    - Score margin between top-1 and top-2 candidates
    - Multi-retriever agreement (Dense + BM25 intersection)
    - Candidate volume and distribution spread
    """

    def __init__(
        self,
        min_coverage_fast: float = 0.50,
        min_coverage_balanced: float = 0.35,
        min_top_score_bm25: float = 2.0,
        confidence_high_threshold: float = 0.85,
        confidence_medium_threshold: float = 0.60,
    ) -> None:
        self.min_coverage_fast = min_coverage_fast
        self.min_coverage_balanced = min_coverage_balanced
        self.min_top_score_bm25 = min_top_score_bm25
        self.high_threshold = confidence_high_threshold
        self.medium_threshold = confidence_medium_threshold

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[Any],
        execution_mode: str = "balanced",
        exact_identifiers: Optional[List[str]] = None,
        retrieval_signals: Optional[Dict[str, Any]] = None,
    ) -> RetrievalConfidence:
        """
        Compute an evidence-based retrieval confidence assessment.
        """
        if not candidates:
            return RetrievalConfidence(
                score=0.0,
                tier="low",
                is_sufficient=False,
                term_coverage=0.0,
                exact_identifier_matches=[],
                missing_identifiers=list(exact_identifiers or []),
                candidate_count=0,
                top_score=0.0,
                score_margin=0.0,
                signals={"term_coverage": 0.0, "score_margin": 0.0, "agreement": 0.0},
                reasons=["No candidates retrieved"],
                reason="No candidates retrieved",
            )

        # 1. Non-stopword query tokens
        q_tokens = [
            w for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query.lower())
            if w not in STOPWORDS
        ]

        top_candidates = candidates[:min(5, len(candidates))]
        top_3_text = " ".join(
            str(getattr(c, "text", "") or getattr(c, "snippet", "") or "")
            for c in top_candidates[:3]
        ).lower()

        # 2. Term coverage
        hits = sum(1 for t in q_tokens if t in top_3_text)
        term_cov = hits / max(1, len(q_tokens))

        # 3. Exact identifier coverage
        req_ids = [str(i).lower().strip() for i in (exact_identifiers or []) if str(i).strip()]
        matched_ids = [i for i in req_ids if i in top_3_text]
        missing_ids = [i for i in req_ids if i not in top_3_text]
        id_ratio = (len(matched_ids) / len(req_ids)) if req_ids else 1.0

        # 4. Score distribution & margin
        scores: List[float] = []
        for c in top_candidates:
            sc = getattr(c, "score", None)
            if sc is None:
                sc = getattr(c, "fusion_score", 0.0)
            try:
                scores.append(float(sc))
            except Exception:
                scores.append(0.0)

        top_score = scores[0] if scores else 0.0
        s2 = scores[1] if len(scores) > 1 else 0.0
        score_margin = max(0.0, top_score - s2)
        relative_margin = (score_margin / max(top_score, 1e-4)) if top_score > 0 else 0.0

        # 5. Dense / Lexical agreement signal
        agreement_count = 0
        for c in top_candidates:
            prov = getattr(c, "provenance_sources", [])
            has_dense = any("dense" in p for p in prov) or getattr(c, "dense_score", None) is not None
            has_bm25 = any("keyword" in p or "bm25" in p for p in prov) or getattr(c, "keyword_score", None) is not None
            if has_dense and has_bm25:
                agreement_count += 1
        agreement_ratio = agreement_count / max(1, len(top_candidates))

        # 6. Synthesize confidence score
        # Base weights calibrated against retrieval signals
        w_cov = 0.35
        w_id = 0.30
        w_margin = 0.15
        w_agree = 0.20

        # Normalized score signals
        norm_margin = min(1.0, relative_margin * 2.0)
        conf_score = (
            w_cov * term_cov
            + w_id * id_ratio
            + w_margin * norm_margin
            + w_agree * agreement_ratio
        )

        # Mode-specific adjustments
        if execution_mode == "fast":
            # For BM25-only fast route, agreement signal is not present
            bm25_norm = min(1.0, top_score / 12.0)
            conf_score = 0.50 * term_cov + 0.35 * id_ratio + 0.15 * bm25_norm

        conf_score = max(0.0, min(1.0, conf_score))

        # 7. Tier classification
        if conf_score >= self.high_threshold:
            tier: Literal["high", "medium", "low"] = "high"
        elif conf_score >= self.medium_threshold:
            tier = "medium"
        else:
            tier = "low"

        # 8. Sufficiency threshold
        threshold = self.min_coverage_fast if execution_mode == "fast" else self.min_coverage_balanced
        is_sufficient = (
            term_cov >= threshold
            and (not req_ids or len(matched_ids) > 0)
            and (execution_mode != "fast" or top_score >= self.min_top_score_bm25 or term_cov >= 0.6)
        )

        reasons = [
            f"Coverage: {term_cov:.1%}",
            f"Identifiers: {len(matched_ids)}/{len(req_ids)}",
            f"Margin: {relative_margin:.2f}",
        ]
        if agreement_count > 0:
            reasons.append(f"Hybrid agreement: {agreement_count}/{len(top_candidates)}")
        reasons.append(f"Tier: {tier.upper()} ({conf_score:.2f})")

        signals = {
            "term_coverage": round(term_cov, 4),
            "exact_id_ratio": round(id_ratio, 4),
            "score_margin": round(score_margin, 4),
            "relative_margin": round(relative_margin, 4),
            "agreement_ratio": round(agreement_ratio, 4),
            "top_score": round(top_score, 4),
            "confidence_score": round(conf_score, 4),
        }

        return RetrievalConfidence(
            score=conf_score,
            tier=tier,
            is_sufficient=is_sufficient,
            term_coverage=term_cov,
            exact_identifier_matches=matched_ids,
            missing_identifiers=missing_ids,
            candidate_count=len(candidates),
            top_score=top_score,
            score_margin=score_margin,
            signals=signals,
            reasons=reasons,
            reason="; ".join(reasons),
        )
