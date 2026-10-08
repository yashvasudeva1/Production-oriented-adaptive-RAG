from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from ..indexing.models import SearchResult


class UnifiedCandidate(BaseModel):
    """Unified candidate chunk representation across all retrieval strategies."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int = 1
    dense_score: float | None = None
    keyword_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    parent_id: str | None = None
    source_locator: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_type: str = "text"
    provenance_sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def reciprocal_rank_fusion(
    candidate_lists: Sequence[Sequence[Any]],
    k: int = 60,
    top_n: int = 20,
    exact_identifiers: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
) -> List[UnifiedCandidate]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion with exact-match protection:
    RRF_score(d) = SUM(weight * (1 / (k + rank(d)))) + exact_bonus(d)
    """
    scores: Dict[str, float] = {}
    best_candidate: Dict[str, Any] = {}
    sources: Dict[str, List[str]] = {}
    dense_scores: Dict[str, float] = {}
    keyword_scores: Dict[str, float] = {}

    if weights:
        list_weights = list(weights)
    elif len(candidate_lists) == 2:
        # Weighted RRF: alpha heavily favors BM25
        # candidate_lists[0] is assumed Dense, candidate_lists[1] is assumed BM25
        alpha = 0.8
        list_weights = [1.0 - alpha, alpha]
    else:
        list_weights = [1.0] * len(candidate_lists)

    for cand_list, w in zip(candidate_lists, list_weights):
        for rank, cand in enumerate(cand_list, start=1):
            cid = getattr(cand, "chunk_id", str(cand))
            rrf_val = w * (1.0 / (k + rank))
            scores[cid] = scores.get(cid, 0.0) + rrf_val

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

    # Apply exact-match protection bonus if exact identifiers are specified
    if exact_identifiers:
        clean_ids = [ident.lower().strip() for ident in exact_identifiers if ident.strip()]
        if clean_ids:
            for cid, item in best_candidate.items():
                item_text = (getattr(item, "text", "") or "").lower()
                matched = sum(1 for ident in clean_ids if ident in item_text)
                if matched > 0:
                    # Boost proportional to matched exact identifiers (max boost equal to Rank 1 RRF value)
                    bonus = (matched / len(clean_ids)) * (1.0 / (k + 1))
                    scores[cid] += bonus

    # Sort candidates by combined RRF score descending
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
