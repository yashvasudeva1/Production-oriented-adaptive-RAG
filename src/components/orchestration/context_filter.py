from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from ..reranker import RerankerResult

logger = logging.getLogger(__name__)


class ContextFilter:
    """
    Applies token budgets, eliminates near-duplicates, and cuts off low-scoring
    chunks to produce a clean, non-redundant context for generation.

    Deduplication uses a combined Jaccard word-set similarity and character bigram
    overlap check, which is more robust than word-only Jaccard for paraphrased or
    partially-overlapping chunks.
    """

    def __init__(
        self,
        max_context_tokens: int = 2048,
        min_rerank_score: float = -10.5,
        similarity_threshold: float = 0.82,
        max_chunks_per_doc: int = 2,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.min_rerank_score = min_rerank_score
        self.similarity_threshold = similarity_threshold
        self.max_chunks_per_doc = max_chunks_per_doc

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _char_bigrams(self, text: str) -> set:
        """Returns the set of character bigrams for a lowercased text string."""
        t = text.lower().replace(" ", "")
        return {t[i: i + 2] for i in range(len(t) - 1)} if len(t) > 1 else set()

    def _is_near_duplicate(self, text_a: str, text_b: str) -> bool:
        """
        Returns True if text_a and text_b are near-duplicates using a combined
        word-level Jaccard score and character bigram overlap.
        The combined check catches both lexically identical and paraphrased duplicates.
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return False
        jaccard_words = len(words_a & words_b) / len(words_a | words_b)
        if jaccard_words >= self.similarity_threshold:
            return True
        # Secondary check: character bigram overlap catches paraphrase duplicates
        bigrams_a = self._char_bigrams(text_a)
        bigrams_b = self._char_bigrams(text_b)
        if bigrams_a and bigrams_b:
            bigram_jaccard = len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)
            if bigram_jaccard >= self.similarity_threshold:
                return True
        return False

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

            # Skip chunks with heavy lexical or bigram overlap to maintain diversity
            if any(self._is_near_duplicate(cand.text, existing.text) for existing in filtered):
                logger.debug(f"Filtering near-duplicate chunk {cand.chunk_id}")
                continue

            # Limit chunks per document to promote multi-document graph traversal
            doc_id = getattr(cand, "document_id", None)
            if doc_id and sum(1 for e in filtered if getattr(e, "document_id", None) == doc_id) >= self.max_chunks_per_doc:
                continue

            # Enforce prompt context token budget
            toks = self._estimate_tokens(cand.text)
            if total_tokens + toks > limit and filtered:
                break

            filtered.append(cand)
            total_tokens += toks

        return filtered
