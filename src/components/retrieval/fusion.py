from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from ..indexing.models import SearchResult


class UnifiedCandidate(BaseModel):
    """Unified candidate chunk representation across all retrieval strategies."""
    chunk_id: str
    document_id: str = "doc_unknown"
    text: str = ""
    score: float = 0.0
    rank: int = 1
    dense_score: float | None = None
    keyword_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None
    parent_id: str | None = None
    source_locator: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_type: str = "text"
    provenance_sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def copy(self, *args, **kwargs) -> UnifiedCandidate:
        return self.model_copy(*args, **kwargs)


def reciprocal_rank_fusion(
    bm25_results: Any = None,
    dense_results: Any = None,
    k: int = 60,
    bm25_weight: float = 0.82,  # Shields our elite lexical precision baseline
    dense_weight: float = 0.18,
    candidate_lists: Optional[Sequence[Sequence[Any]]] = None,
    top_n: Optional[int] = None,
    exact_identifiers: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> List[UnifiedCandidate]:
    """
    CPU-Optimized Weighted Reciprocal Rank Fusion.
    Safeguards high-performing keyword precision against dense space noise.
    Supports both direct (bm25_results, dense_results) pairs and multi-candidate lists.
    """
    resolved_lists: List[Sequence[Any]] = []
    list_weights: List[float] = []

    # Handle various calling conventions
    if candidate_lists is not None:
        resolved_lists = list(candidate_lists)
    elif (
        isinstance(bm25_results, (list, tuple))
        and len(bm25_results) > 0
        and isinstance(bm25_results[0], (list, tuple))
    ):
        resolved_lists = list(bm25_results)
        if isinstance(dense_results, (int, float)):
            k = int(dense_results)
            dense_results = None
    elif bm25_results is not None and dense_results is not None and isinstance(dense_results, (list, tuple, Sequence)) and not isinstance(dense_results, (str, bytes)):
        resolved_lists = [bm25_results, dense_results]
        list_weights = [bm25_weight, dense_weight]
    elif bm25_results is not None:
        resolved_lists = [bm25_results]
        list_weights = [1.0]

    # Resolve weights if not explicitly assigned
    if weights is not None:
        list_weights = list(weights)
    elif not list_weights:
        if len(resolved_lists) == 2:
            l0 = resolved_lists[0]
            l1 = resolved_lists[1]
            def _is_dense(lst):
                for item in lst[:3]:
                    ret_name = str(getattr(item, "retriever_name", "") or (item.get("retriever_name", "") if isinstance(item, dict) else "")).lower()
                    if "dense" in ret_name:
                        return True
                return False
            def _is_bm25(lst):
                for item in lst[:3]:
                    ret_name = str(getattr(item, "retriever_name", "") or (item.get("retriever_name", "") if isinstance(item, dict) else "")).lower()
                    if "bm25" in ret_name or "keyword" in ret_name:
                        return True
                return False

            if _is_dense(l0) and _is_bm25(l1):
                list_weights = [dense_weight, bm25_weight]
            elif _is_bm25(l0) and _is_dense(l1):
                list_weights = [bm25_weight, dense_weight]
            else:
                list_weights = [bm25_weight, dense_weight]
        else:
            list_weights = [1.0] * len(resolved_lists)

    scores: Dict[str, float] = {}
    doc_mapping: Dict[str, Any] = {}
    sources: Dict[str, List[str]] = {}
    dense_scores: Dict[str, float] = {}
    keyword_scores: Dict[str, float] = {}

    for cand_list, w in zip(resolved_lists, list_weights):
        for rank, doc in enumerate(cand_list):
            if isinstance(doc, (dict, SearchResult, UnifiedCandidate)) or hasattr(doc, "__getitem__"):
                doc_id = doc["chunk_id"] if "chunk_id" in doc else getattr(doc, "chunk_id", str(doc))
            else:
                doc_id = getattr(doc, "chunk_id", str(doc))

            if doc_id not in doc_mapping:
                doc_mapping[doc_id] = doc
                sources[doc_id] = []

            # Weighted RRF score formula: w * (1 / (k + rank + 1))
            rrf_val = w * (1.0 / (k + rank + 1))
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_val

            # Track sources and retriever scores
            ret_name = getattr(doc, "retriever_name", None) or (doc.get("retriever_name") if isinstance(doc, dict) else "") or ""
            prov = getattr(doc, "provenance_sources", None) or (doc.get("provenance_sources") if isinstance(doc, dict) else None) or []
            for p in prov:
                if p not in sources[doc_id]:
                    sources[doc_id].append(p)
            if ret_name and ret_name not in sources[doc_id]:
                sources[doc_id].append(ret_name)

            if "dense" in str(ret_name).lower() or any("dense" in str(p).lower() for p in prov) or w == dense_weight:
                if "dense" not in sources[doc_id]:
                    sources[doc_id].append("dense")
                sc = getattr(doc, "score", None) or (doc.get("score") if isinstance(doc, dict) else None)
                if sc is not None:
                    dense_scores[doc_id] = float(sc)
            if "keyword" in str(ret_name).lower() or "bm25" in str(ret_name).lower() or any("bm25" in str(p).lower() for p in prov) or w == bm25_weight:
                if "bm25" not in sources[doc_id]:
                    sources[doc_id].append("bm25")
                sc = getattr(doc, "score", None) or (doc.get("score") if isinstance(doc, dict) else None)
                if sc is not None:
                    keyword_scores[doc_id] = float(sc)

    # Apply exact match protection bonus if specified
    if exact_identifiers:
        clean_ids = [ident.lower().strip() for ident in exact_identifiers if ident.strip()]
        if clean_ids:
            for cid, item in doc_mapping.items():
                item_text = (getattr(item, "text", "") or (item.get("text", "") if isinstance(item, dict) else "") or "").lower()
                matched = sum(1 for ident in clean_ids if ident in item_text)
                if matched > 0:
                    bonus = (matched / len(clean_ids)) * (1.0 / (k + 1))
                    scores[cid] += bonus

    # Efficient serialization loop
    fused_results: List[UnifiedCandidate] = []
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None and top_n > 0:
        sorted_items = sorted_items[:top_n]

    for rank, (doc_id, final_score) in enumerate(sorted_items, start=1):
        raw_doc = doc_mapping[doc_id]
        cand = UnifiedCandidate(
            chunk_id=str(doc_id),
            document_id=getattr(raw_doc, "document_id", None) or (raw_doc.get("document_id", "doc_unknown") if isinstance(raw_doc, dict) else "doc_unknown"),
            text=getattr(raw_doc, "text", None) or (raw_doc.get("text", "") if isinstance(raw_doc, dict) else ""),
            score=final_score,
            rank=rank,
            dense_score=dense_scores.get(doc_id),
            keyword_score=keyword_scores.get(doc_id),
            fusion_score=final_score,
            rrf_score=final_score,
            parent_id=getattr(raw_doc, "parent_id", None) or (raw_doc.get("parent_id") if isinstance(raw_doc, dict) else None),
            source_locator=getattr(raw_doc, "source_locator", None) or (raw_doc.get("source_locator") if isinstance(raw_doc, dict) else None),
            page=getattr(raw_doc, "page", None) or (raw_doc.get("page") if isinstance(raw_doc, dict) else None),
            section=getattr(raw_doc, "section", None) or (raw_doc.get("section") if isinstance(raw_doc, dict) else None),
            chunk_type=getattr(raw_doc, "chunk_type", "text") or (raw_doc.get("chunk_type", "text") if isinstance(raw_doc, dict) else "text"),
            provenance_sources=list(sources.get(doc_id, [])),
            metadata=getattr(raw_doc, "metadata", None) or (raw_doc.get("metadata", {}) if isinstance(raw_doc, dict) else {}),
        )
        fused_results.append(cand)

    return fused_results


def weighted_rrf(
    candidate_lists: Sequence[Sequence[Any]],
    weights: Sequence[float],
    k: int = 60,
    top_n: int = 20,
    exact_identifiers: Optional[Sequence[str]] = None,
) -> List[UnifiedCandidate]:
    """Weighted Reciprocal Rank Fusion delegating to reciprocal_rank_fusion with explicit weights."""
    return reciprocal_rank_fusion(
        candidate_lists=candidate_lists,
        k=k,
        top_n=top_n,
        exact_identifiers=exact_identifiers,
        weights=weights,
    )


def score_normalized_fusion(
    candidate_lists: Sequence[Sequence[Any]],
    weights: Optional[Sequence[float]] = None,
    top_n: int = 20,
    exact_identifiers: Optional[Sequence[str]] = None,
) -> List[UnifiedCandidate]:
    """
    Combines candidate lists by min-max normalizing raw scores in each list
    and taking a weighted linear combination:
    score(d) = SUM(w_i * norm_score_i(d)) + exact_bonus(d)
    """
    scores: Dict[str, float] = {}
    best_candidate: Dict[str, Any] = {}
    sources: Dict[str, List[str]] = {}
    dense_scores: Dict[str, float] = {}
    keyword_scores: Dict[str, float] = {}

    list_weights = list(weights) if weights else [1.0] * len(candidate_lists)

    for cand_list, w in zip(candidate_lists, list_weights):
        if not cand_list:
            continue
        raw_vals = [float(getattr(c, "score", 0.0)) for c in cand_list]
        min_v = min(raw_vals)
        max_v = max(raw_vals)
        spread = max_v - min_v

        for cand in cand_list:
            cid = getattr(cand, "chunk_id", str(cand))
            raw_s = float(getattr(cand, "score", 0.0))
            norm_s = (raw_s - min_v) / (spread + 1e-9) if spread > 1e-6 else 1.0

            scores[cid] = scores.get(cid, 0.0) + (w * norm_s)

            if cid not in best_candidate:
                best_candidate[cid] = cand
                sources[cid] = []

            ret_name = getattr(cand, "retriever_name", "") or "unknown"
            if ret_name and ret_name not in sources[cid]:
                sources[cid].append(ret_name)

            if "dense" in ret_name:
                dense_scores[cid] = getattr(cand, "score", 0.0)
            elif "keyword" in ret_name or "bm25" in ret_name:
                keyword_scores[cid] = getattr(cand, "score", 0.0)

    if exact_identifiers:
        clean_ids = [ident.lower().strip() for ident in exact_identifiers if ident.strip()]
        if clean_ids:
            for cid, item in best_candidate.items():
                item_text = (getattr(item, "text", "") or "").lower()
                matched = sum(1 for ident in clean_ids if ident in item_text)
                if matched > 0:
                    bonus = (matched / len(clean_ids)) * 0.2
                    scores[cid] += bonus

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    results: List[UnifiedCandidate] = []
    for rank, cid in enumerate(sorted_ids[:top_n], start=1):
        item = best_candidate[cid]
        results.append(
            UnifiedCandidate(
                chunk_id=getattr(item, "chunk_id", str(cid)),
                document_id=getattr(item, "document_id", "doc_unknown"),
                text=getattr(item, "text", ""),
                score=scores[cid],
                rank=rank,
                dense_score=dense_scores.get(cid),
                keyword_score=keyword_scores.get(cid),
                fusion_score=scores[cid],
                parent_id=getattr(item, "parent_id", None),
                source_locator=getattr(item, "source_locator", None),
                page=getattr(item, "page", None),
                section=getattr(item, "section", None),
                chunk_type=getattr(item, "chunk_type", "text"),
                provenance_sources=sources.get(cid, []),
                metadata=getattr(item, "metadata", {}),
            )
        )

    return results


def fuse_candidates(
    candidate_lists: Sequence[Sequence[Any]],
    method: str = "rrf",
    weights: Optional[Sequence[float]] = None,
    k: int = 60,
    top_n: int = 20,
    exact_identifiers: Optional[Sequence[str]] = None,
) -> List[UnifiedCandidate]:
    """Unified dispatcher for multi-strategy candidate rank fusion."""
    norm_method = (method or "rrf").lower().strip()
    if norm_method in ("score_normalized", "normalized", "linear"):
        return score_normalized_fusion(
            candidate_lists=candidate_lists,
            weights=weights,
            top_n=top_n,
            exact_identifiers=exact_identifiers,
        )
    elif norm_method in ("weighted_rrf", "weighted") and weights is not None:
        return weighted_rrf(
            candidate_lists=candidate_lists,
            weights=weights,
            k=k,
            top_n=top_n,
            exact_identifiers=exact_identifiers,
        )
    return reciprocal_rank_fusion(
        candidate_lists=candidate_lists,
        k=k,
        top_n=top_n,
        exact_identifiers=exact_identifiers,
        weights=weights,
    )
