from __future__ import annotations

import logging
from typing import List, Sequence

from ..reranker import RerankerResult

logger = logging.getLogger(__name__)


class ContextFilter:
    """
    Applies token budgets, eliminates near-duplicates, and cuts off low-scoring
    chunks to produce a clean, non-redundant context for generation.
    """

    def __init__(
        self,
        max_context_tokens: int = 2048,
        min_rerank_score: float = -5.0,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.min_rerank_score = min_rerank_score
        self.similarity_threshold = similarity_threshold

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _is_near_duplicate(self, text_a: str, text_b: str) -> bool:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return False
        jaccard = len(words_a & words_b) / len(words_a | words_b)
        return jaccard >= self.similarity_threshold

    def filter_context(
        self,
        candidates: Sequence[RerankerResult],
        max_tokens: Optional[int] = None,
    ) -> List[RerankerResult]:
        if not candidates:
            return []

        limit = max_tokens or self.max_context_tokens
        filtered: List[RerankerResult] = []
        total_tokens = 0

        for cand in candidates:
            # Drop low-confidence candidates below relevance cutoff
            if cand.rerank_score < self.min_rerank_score:
                continue

            # Skip chunks with heavy lexical overlap to maintain diversity
            if any(self._is_near_duplicate(cand.text, existing.text) for existing in filtered):
                logger.debug(f"Filtering near-duplicate chunk {cand.chunk_id}")
                continue

            # Enforce prompt context token budget
            toks = self._estimate_tokens(cand.text)
            if total_tokens + toks > limit and filtered:
                break

            filtered.append(cand)
            total_tokens += toks

        return filtered
