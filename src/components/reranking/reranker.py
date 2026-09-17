from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field

from ..retrieval.fusion import UnifiedCandidate

logger = logging.getLogger(__name__)


def _get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


class RerankerResult(BaseModel):
    """Normalized output from reranker preserving full document and score provenance."""
    chunk_id: str
    document_id: str
    text: str
    score: float = 0.0
    rerank_score: float
    rank: int = 1
    retrieval_score: Optional[float] = None
    dense_score: Optional[float] = None
    keyword_score: Optional[float] = None
    fusion_score: Optional[float] = None
    parent_id: Optional[str] = None
    source_locator: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_type: str = "text"
    provenance_sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrossEncoderReranker:
    """
    Production cross-encoder reranker with support for conditional bypass,
    candidate deduplication, and complete provenance retention.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        default_top_k: int = 5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.default_top_k = default_top_k
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Loaded CrossEncoder: {self.model_name}")
            except Exception as exc:
                logger.warning(
                    f"Could not load CrossEncoder '{self.model_name}': {exc}. Using lexical fallback."
                )
                self._model = False

    def _fallback_score(self, query: str, text: str) -> float:
        """Lexical Jaccard/overlap fallback score between query and candidate text."""
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        if not q_words or not t_words:
            return 0.0
        intersection = q_words & t_words
        return len(intersection) / math.sqrt(len(q_words) * len(t_words))

    def should_rerank(
        self,
        query_type: str = "general",
        execution_mode: str = "balanced",
        candidates: Optional[Sequence[Any]] = None,
    ) -> bool:
        """Determines whether reranking adds value or if candidates are sufficiently strong."""
        if execution_mode == "fast":
            return False
        if not candidates:
            return False
        if len(candidates) <= 1:
            return False
        # If fact query has a candidate with very high dominant score, skip rerank
        if query_type in ("fact", "technical_exact") and candidates:
            top_score = _get_field(candidates[0], "fusion_score", 0.0) or _get_field(candidates[0], "score", 0.0)
            if float(top_score) >= 0.03:
                return False
        return True

    def bypass_rerank(
        self,
        candidates: Sequence[UnifiedCandidate | Any],
        top_k: Optional[int] = None,
    ) -> List[RerankerResult]:
        """Convert candidates directly to RerankerResult without cross-encoder inference."""
        k = top_k or self.default_top_k
        unique_cands: Dict[str, Any] = {}
        for c in candidates:
            cid = _get_field(c, "chunk_id", None)
            if cid and cid not in unique_cands:
                unique_cands[cid] = c

        cand_list = list(unique_cands.values())[:k]
        results: List[RerankerResult] = []
        for rank, c in enumerate(cand_list, start=1):
            score = float(_get_field(c, "fusion_score", 0.0) or _get_field(c, "score", 0.0))
            results.append(
                RerankerResult(
                    chunk_id=_get_field(c, "chunk_id", ""),
                    document_id=_get_field(c, "document_id", ""),
                    text=_get_field(c, "text", ""),
                    score=score,
                    rerank_score=score,
                    rank=rank,
                    retrieval_score=score,
                    dense_score=_get_field(c, "dense_score", None),
                    keyword_score=_get_field(c, "keyword_score", None),
                    fusion_score=_get_field(c, "fusion_score", None),
                    parent_id=_get_field(c, "parent_id", None),
                    source_locator=_get_field(c, "source_locator", None),
                    page=_get_field(c, "page", None),
                    section=_get_field(c, "section", None),
                    metadata=_get_field(c, "metadata", {}) or {},
                )
            )
        return results

    def rerank(
        self,
        query: str,
        candidates: Sequence[UnifiedCandidate | Any],
        top_k: Optional[int] = None,
        force: bool = False,
    ) -> List[RerankerResult]:
        if not candidates:
            return []

        k = top_k or self.default_top_k
        if k <= 0:
            return []

        # Deduplicate incoming candidates from multiple retrieval paths
        unique_cands: Dict[str, Any] = {}
        for c in candidates:
            cid = _get_field(c, "chunk_id", None)
            if cid and cid not in unique_cands:
                unique_cands[cid] = c

        cand_list = list(unique_cands.values())
        if not cand_list:
            return []

        # Score query-document pairs with cross-encoder (or lexical overlap if offline)
        self._load_model()
        pairs = [[query, _get_field(c, "text", "")] for c in cand_list]

        if self._model and self._model is not False:
            try:
                scores = self._model.predict(pairs)
            except Exception as exc:
                logger.error(f"Cross-encoder predict failed: {exc}, using fallback.")
                scores = [self._fallback_score(query, p[1]) for p in pairs]
        else:
            scores = [self._fallback_score(query, p[1]) for p in pairs]

        scored_results: List[RerankerResult] = []
        for cand, score in zip(cand_list, scores):
            text = _get_field(cand, "text", "")
            cid = _get_field(cand, "chunk_id", "")
            did = _get_field(cand, "document_id", "")
            parent_id = _get_field(cand, "parent_id", None)
            locator = _get_field(cand, "source_locator", None)
            page = _get_field(cand, "page", None)
            section = _get_field(cand, "section", None)
            ctype = _get_field(cand, "chunk_type", "text")
            prov = _get_field(cand, "provenance_sources", [])
            meta = _get_field(cand, "metadata", {})

            retrieval_score = _get_field(cand, "score", None)
            dense_s = _get_field(cand, "dense_score", None)
            key_s = _get_field(cand, "keyword_score", None)
            fus_s = _get_field(cand, "fusion_score", None)

            scored_results.append(
                RerankerResult(
                    chunk_id=cid,
                    document_id=did,
                    text=text,
                    rerank_score=float(score),
                    retrieval_score=float(retrieval_score) if retrieval_score is not None else None,
                    dense_score=float(dense_s) if dense_s is not None else None,
                    keyword_score=float(key_s) if key_s is not None else None,
                    fusion_score=float(fus_s) if fus_s is not None else None,
                    parent_id=parent_id,
                    source_locator=locator,
                    page=page,
                    section=section,
                    chunk_type=ctype,
                    provenance_sources=prov,
                    metadata=meta,
                )
            )

        scored_results.sort(key=lambda r: r.rerank_score, reverse=True)

        for rank, r in enumerate(scored_results[:k], start=1):
            r.rank = rank

        return scored_results[:k]
