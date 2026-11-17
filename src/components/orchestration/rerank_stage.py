from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Sequence

from ..reranker import CrossEncoderReranker, RerankerResult
from .contracts import RerankStageResult

logger = logging.getLogger(__name__)


class RerankStage:
    """
    Independent Conditional Reranking Stage.
    Implements Top-1 Consensus Guard and Tiered Micro-Reranking.

    Policy:
    - Consensus Bypass: High confidence (>= 0.82) + Top-1 agreement between Dense and BM25
      skips the expensive cross-encoder (~150ms latency savings on CPU).
    - Tiered Micro-Rerank: Borderline confidence (0.65 <= Conf < 0.82) reranks top-6 candidates.
    - Full Rerank: Low confidence (< 0.65) or forced modes reranks top-10 candidates.
    """

    def __init__(self, reranker: CrossEncoderReranker, enable_consensus_guard: bool = True) -> None:
        self.reranker = reranker
        self.enable_consensus_guard = enable_consensus_guard

    def execute_rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        mode: str,
        confidence_score: float,
        top_bm25_id: Optional[str] = None,
        top_dense_id: Optional[str] = None,
        exact_identifiers: Optional[List[str]] = None,
        rerank_top_k: int = 5,
        default_top_k: int = 5,
        enable_consensus_guard: Optional[bool] = None,
    ) -> RerankStageResult:
        if not candidates:
            return RerankStageResult(candidates=[], did_rerank=False, can_bypass=True, rerank_latency_ms=0.0)

        exact_ids = exact_identifiers or []
        t0 = time.perf_counter()
        use_guard = self.enable_consensus_guard if enable_consensus_guard is None else enable_consensus_guard

        # Top-1 Consensus Guard Evaluation
        top1_agreement = (top_bm25_id is not None and top_bm25_id == top_dense_id)
        has_unique_exact_id = False
        if exact_ids and candidates:
            d1_text = getattr(candidates[0], "text", "").lower()
            for eid in exact_ids:
                eid_lower = eid.lower()
                if eid_lower in d1_text:
                    in_others = any(eid_lower in getattr(c, "text", "").lower() for c in candidates[1:5])
                    if not in_others:
                        has_unique_exact_id = True
                        break

        can_full_bypass = (
            use_guard
            and confidence_score >= 0.82
            and (top1_agreement or has_unique_exact_id)
        )

        if mode == "hybrid_rerank":
            should_rerank = True
            rerank_pool = list(candidates)
        elif mode in ("fast", "hybrid", "bm25"):
            should_rerank = False
            rerank_pool = list(candidates)
        elif can_full_bypass:
            should_rerank = False
            rerank_pool = list(candidates)
        else:
            should_rerank = True
            # Tiered Micro-Rerank for Borderline Band (0.65 <= conf.score < 0.82)
            if confidence_score >= 0.65:
                rerank_pool = list(candidates[:6])
            else:
                rerank_pool = list(candidates[:10])

        if should_rerank:
            reranked = self.reranker.rerank(
                query=query,
                candidates=rerank_pool,
                top_k=rerank_top_k,
                max_candidates=len(rerank_pool),
            )
            did_rerank = True
        else:
            reranked = self.reranker.bypass_rerank(list(candidates), top_k=default_top_k)
            did_rerank = False

        latency_ms = (time.perf_counter() - t0) * 1000

        return RerankStageResult(
            candidates=reranked,
            did_rerank=did_rerank,
            can_bypass=can_full_bypass,
            rerank_latency_ms=latency_ms,
        )
