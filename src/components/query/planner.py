from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..metadata_filtering import MetadataFilter
from ..query_metadata_extraction import QueryMetadata
from .categorization import AdaptiveQueryCategorizer
from .decomposition import decompose_query
from .metadata import AdaptiveQueryMetadataExtractor
from .models import Answerability, ExecutionMode, QueryComplexity, QueryPlan, QueryType
from .normalization import normalize_query
from .signals import QuerySignals, extract_query_signals

logger = logging.getLogger(__name__)


class QueryPlanner:
    """
    Tiered Query Planner and Cascade Router.
    Implements:
    - Level 0: Query Normalization
    - Level 1: Deterministic Query Signals (sub-millisecond)
    - Level 2: Execution Mode Routing (FAST, BALANCED, DEEP)
    """

    def __init__(
        self,
        categorizer: Optional[AdaptiveQueryCategorizer] = None,
        metadata_extractor: Optional[AdaptiveQueryMetadataExtractor] = None,
        metadata_filter: Optional[MetadataFilter] = None,
        document_store: Optional[Any] = None,
    ) -> None:
        self.categorizer = categorizer or AdaptiveQueryCategorizer()
        self.metadata_extractor = metadata_extractor or AdaptiveQueryMetadataExtractor()
        self.document_store = document_store
        try:
            self.metadata_filter = metadata_filter or MetadataFilter()
        except Exception as exc:
            logger.warning(f"Could not load default MetadataFilter: {exc}")
            self.metadata_filter = None

    def plan(
        self,
        query: str,
        explicit_document_ids: Optional[List[str]] = None,
        forced_mode: Optional[ExecutionMode] = None,
    ) -> QueryPlan:
        # Level 0: Fast Normalization (<0.05 ms)
        norm_query = normalize_query(query)

        # Level 1: Sub-millisecond Deterministic Signals (<0.2 ms)
        signals = extract_query_signals(query, norm_query)

        # Mode selection based on forced override or deterministic signals
        if forced_mode in ("fast", "balanced", "deep"):
            mode: ExecutionMode = forced_mode
        else:
            mode = signals.initial_route

        # Fast path bypasses heavy metadata filtering & decomposition
        candidate_ids: List[str] = list(explicit_document_ids or [])
        retrieval_signals: Dict[str, Any] = {
            "signals": signals,
            "exact_identifiers": signals.exact_identifiers,
            "quoted_phrases": signals.quoted_phrases,
            "code_symbols": signals.code_symbols,
        }
        extracted_meta: Dict[str, Any] = {}
        from ..retrieval.filters import RetrievalFilter, FilterCondition

        retrieval_filter = RetrievalFilter()
        if candidate_ids:
            retrieval_filter.document_ids = list(candidate_ids)

        # Run metadata filtering when constraints are detected
        if self.metadata_extractor:
            extracted_meta = self.metadata_extractor.extract(norm_query)

            # Build canonical hard filters directly on metadata fields
            if extracted_meta.get("document_type"):
                retrieval_filter.add_hard_filter("document_type", extracted_meta["document_type"])
            if extracted_meta.get("department"):
                retrieval_filter.add_hard_filter("department", extracted_meta["department"])
            if extracted_meta.get("organizations"):
                orgs = extracted_meta["organizations"]
                retrieval_filter.add_hard_filter("organizations", orgs if isinstance(orgs, list) else [orgs], operator="in")
            if extracted_meta.get("dates"):
                dts = extracted_meta["dates"]
                retrieval_filter.add_hard_filter("dates", dts if isinstance(dts, list) else [dts], operator="in")
            if extracted_meta.get("locations"):
                locs = extracted_meta["locations"]
                retrieval_filter.add_hard_filter("locations", locs if isinstance(locs, list) else [locs], operator="in")

            # Soft preferences / retrieval hints
            if extracted_meta.get("topics"):
                retrieval_signals["topics"] = extracted_meta["topics"]
            if signals.exact_identifiers:
                retrieval_filter.retrieval_hints.technical_identifiers.extend(signals.exact_identifiers)
            if signals.quoted_phrases:
                retrieval_filter.retrieval_hints.quoted_phrases.extend(signals.quoted_phrases)

            # Legacy metadata filter compatibility for backward-compatible tests
            if mode in ("balanced", "deep") and self.metadata_filter and not candidate_ids:
                if extracted_meta.get("document_type") or extracted_meta.get("organizations") or extracted_meta.get("dates"):
                    try:
                        qm = QueryMetadata(
                            document_type=extracted_meta.get("document_type", ""),
                            organizations=extracted_meta.get("organizations", []),
                            locations=extracted_meta.get("locations", []),
                            dates=extracted_meta.get("dates", []),
                            department=extracted_meta.get("department", ""),
                            topics=extracted_meta.get("topics", []),
                        )
                        filter_res = self.metadata_filter.filter(query_metadata=qm)
                        if filter_res.applied_filters and filter_res.document_ids:
                            candidate_ids = list(set(filter_res.document_ids))
                            if not retrieval_filter.document_ids:
                                retrieval_filter.document_ids = candidate_ids
                        retrieval_signals.update(filter_res.retrieval_signals)
                    except Exception as exc:
                        logger.warning(f"Legacy metadata filter fallback skipped: {exc}")

        # Sub-query decomposition for deep mode
        sub_queries: List[str] = []
        if mode == "deep":
            sub_queries = decompose_query(norm_query)

        # Configure retrieval parameters by mode
        if mode == "fast":
            # Cheap first: start with BM25 (or small hybrid) with zero reranking
            req_dense = False
            req_keyword = True
            req_hybrid = False
            req_parent_child = False
            req_multi_query = False
            req_decomposition = False
            req_rerank = False
            top_k = 6
            top_k_dense = 0
            top_k_keyword = 6
            rerank_top_k = 0
            context_budget = 1200
        elif mode == "deep":
            req_dense = True
            req_keyword = True
            req_hybrid = True
            req_parent_child = True
            req_multi_query = True
            req_decomposition = len(sub_queries) > 1
            req_rerank = True
            top_k = 15
            top_k_dense = 15
            top_k_keyword = 12
            rerank_top_k = 8
            context_budget = 3000
        else:
            # Balanced mode: parallel BM25 + dense, RRF fusion, conditional rerank
            req_dense = True
            req_keyword = True
            req_hybrid = True
            req_parent_child = (signals.intent in ("detailed", "procedural"))
            req_multi_query = False
            req_decomposition = False
            req_rerank = True
            top_k = 10
            top_k_dense = 10
            top_k_keyword = 10
            rerank_top_k = 5
            context_budget = 2048

        has_meta_filter = bool(candidate_ids)
        q_type: QueryType = (
            "comparison" if signals.intent == "comparison"
            else "technical_exact" if signals.intent == "technical_exact"
            else "fact" if signals.intent == "fact"
            else "detailed" if signals.intent in ("detailed", "procedural")
            else "general"
        )
        if has_meta_filter and q_type not in ("technical_exact", "comparison"):
            q_type = "metadata_constrained"

        return QueryPlan(
            query=query,
            normalized_query=norm_query,
            query_type=q_type,
            complexity=signals.complexity,
            execution_mode=mode,
            answerability="supported",
            sub_queries=sub_queries if req_decomposition else [],
            candidate_document_ids=candidate_ids,
            metadata_filters=extracted_meta,
            retrieval_filter=retrieval_filter,
            retrieval_signals=retrieval_signals,
            requires_dense=req_dense,
            requires_keyword=req_keyword,
            requires_dense_search=req_dense,
            requires_keyword_search=req_keyword,
            requires_hybrid=req_hybrid,
            requires_parent_child=req_parent_child,
            requires_multi_query=req_multi_query,
            requires_decomposition=req_decomposition,
            requires_metadata_filter=has_meta_filter or not retrieval_filter.is_empty(),
            requires_reranking=req_rerank,
            top_k=top_k,
            top_k_dense=top_k_dense,
            top_k_keyword=top_k_keyword,
            rerank_top_k=rerank_top_k,
            context_budget=context_budget,
            confidence=0.90 if mode == "fast" else 0.80,
            reasoning=f"Route: {mode.upper()} via signals (intent={signals.intent}, complexity={signals.complexity})",
        )
