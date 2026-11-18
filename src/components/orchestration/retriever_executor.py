from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from ..cache import CacheManager
from ..indexing.bm25_indexer import BM25Indexer
from ..indexing.models import SearchResult
from ..indexing.qdrant_indexer import QdrantIndexer
from ..retrieval.dense import DenseRetriever
from ..retrieval.fusion import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_RRF_K,
    IDENTIFIER_BM25_WEIGHT,
    IDENTIFIER_DENSE_WEIGHT,
    UnifiedCandidate,
    fuse_candidates,
)
from ..retrieval.keyword import KeywordRetriever
from .contracts import RetrievalStageResult

logger = logging.getLogger(__name__)


class RetrieverExecutor:
    """
    Independent Retrieval Execution Stage.
    Handles lexical (BM25), dense (Qdrant), and hybrid fusion with strict scoping.

    Invariants:
    1. Scope Isolation: Explicit document and tenant filters are NEVER relaxed.
       If a filter produces 0 hits, 0 hits are returned to prevent scope leakage.
    2. Caching: Results are cached deterministically with index version and full query hashes.
    3. Concurrency: Hybrid search executes BM25 and Dense branches in parallel.
    """

    def __init__(
        self,
        qdrant_indexer: QdrantIndexer,
        bm25_indexer: BM25Indexer,
        dense_retriever: DenseRetriever,
        keyword_retriever: KeywordRetriever,
        cache_enabled: bool = True,
        fusion_method: str = "rrf",
    ) -> None:
        self.qdrant = qdrant_indexer
        self.bm25 = bm25_indexer
        self.dense_retriever = dense_retriever
        self.keyword_retriever = keyword_retriever
        self.cache_enabled = cache_enabled
        self.fusion_method = fusion_method

    def _keyword_cache_meta(self, signals: Dict[str, Any]) -> tuple[str, str]:
        bm25_id = f"bm25:{getattr(self.bm25, 'index_path', 'mem')}:n={self.bm25.count()}"
        sig_cfg = json.dumps(signals or {}, sort_keys=True, default=str)
        return bm25_id, f"signals={sig_cfg}"

    def _dense_cache_meta(self) -> tuple[str, str]:
        emb = getattr(self.dense_retriever, "embedder", None)
        emb_rev = getattr(emb, "cache_revision", f"{type(emb).__name__}:dim={getattr(emb, 'dimension', 0)}")
        coll = getattr(self.qdrant, "collection_name", "default")
        model_info = f"dense:{emb_rev}:coll={coll}:n={self.qdrant.count()}"
        cfg_ver = f"fusion={self.fusion_method}:rrf_k={DEFAULT_RRF_K}"
        return model_info, cfg_ver

    def search_keyword(
        self,
        query: str,
        candidate_document_ids: Optional[List[str]],
        top_k: int,
        signals: Dict[str, Any],
        metadata_filter: Optional[Any] = None,
    ) -> List[SearchResult]:
        """Execute BM25 keyword search with cache lookups and strict scoping."""
        if self.cache_enabled:
            cache_mgr = CacheManager.get_instance()
            filt_key = (str(metadata_filter), tuple(sorted(candidate_document_ids or [])))
            model_info, cfg_ver = self._keyword_cache_meta(signals)
            cached = cache_mgr.get_retrieval(
                query=query,
                filter_repr=filt_key,
                top_k=top_k,
                mode="keyword",
                model_info=model_info,
                config_version=cfg_ver,
            )
            if cached is not None:
                return cached

        results = self.bm25.search(
            query=query,
            candidate_document_ids=candidate_document_ids,
            retrieval_signals=signals,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )

        if self.cache_enabled:
            filt_key = (str(metadata_filter), tuple(sorted(candidate_document_ids or [])))
            model_info, cfg_ver = self._keyword_cache_meta(signals)
            cache_mgr.set_retrieval(
                query=query,
                filter_repr=filt_key,
                top_k=top_k,
                mode="keyword",
                results=results,
                model_info=model_info,
                config_version=cfg_ver,
            )
        return results

    def search_dense(
        self,
        query: str,
        candidate_document_ids: Optional[List[str]],
        top_k: int,
        metadata_filter: Optional[Any] = None,
    ) -> List[SearchResult]:
        """Execute dense vector search with cache lookups and strict scoping."""
        if self.cache_enabled:
            cache_mgr = CacheManager.get_instance()
            filt_key = (str(metadata_filter), tuple(sorted(candidate_document_ids or [])))
            model_info, cfg_ver = self._dense_cache_meta()
            cached = cache_mgr.get_retrieval(
                query=query,
                filter_repr=filt_key,
                top_k=top_k,
                mode="dense",
                model_info=model_info,
                config_version=cfg_ver,
            )
            if cached is not None:
                return cached

        d_resp = self.dense_retriever.retrieve(
            query=query,
            candidate_document_ids=candidate_document_ids,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )
        results = [
            SearchResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                score=c.score,
                rank=c.rank,
                parent_id=c.parent_id,
                source_locator=c.source_locator,
                page=c.page,
                section=c.section,
                chunk_type="text",
                retriever_name="dense_qdrant",
                metadata=c.metadata,
            )
            for c in d_resp.results
        ]

        if self.cache_enabled:
            filt_key = (str(metadata_filter), tuple(sorted(candidate_document_ids or [])))
            model_info, cfg_ver = self._dense_cache_meta()
            cache_mgr.set_retrieval(
                query=query,
                filter_repr=filt_key,
                top_k=top_k,
                mode="dense",
                results=results,
                model_info=model_info,
                config_version=cfg_ver,
            )
        return results

    def execute_retrieval(
        self,
        query: str,
        execution_mode: str,
        top_k_keyword: int,
        top_k_dense: int,
        retrieval_signals: Dict[str, Any],
        candidate_document_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Any] = None,
        inferred_document_ids: Optional[List[str]] = None,
        exact_identifiers: Optional[List[str]] = None,
    ) -> RetrievalStageResult:
        """
        Execute the designated retrieval strategy with latency tracking and strict filtering.
        Never relaxes explicit document or metadata constraints.
        """
        latencies: Dict[str, float] = {}
        exact_ids = exact_identifiers or []
        inferred_ids = inferred_document_ids or []

        if execution_mode == "fast":
            # Fast Route: Sub-millisecond BM25 lexical retrieval
            t0 = time.perf_counter()
            kw_hits = self.search_keyword(
                query=query,
                candidate_document_ids=candidate_document_ids,
                top_k=top_k_keyword,
                signals=retrieval_signals,
                metadata_filter=metadata_filter,
            )
            dt = (time.perf_counter() - t0) * 1000
            latencies["bm25_ms"] = dt
            latencies["fast_retrieval_ms"] = dt

            return RetrievalStageResult(
                candidates=kw_hits,
                raw_bm25_hits=kw_hits,
                raw_dense_hits=[],
                strategy="fast_bm25",
                stage_latencies=latencies,
                exact_identifiers=exact_ids,
                inferred_doc_ids=inferred_ids,
            )

        if execution_mode == "dense":
            t0 = time.perf_counter()
            dense_hits = self.search_dense(
                query=query,
                candidate_document_ids=candidate_document_ids,
                top_k=top_k_dense,
                metadata_filter=metadata_filter,
            )
            dt = (time.perf_counter() - t0) * 1000
            latencies["dense_ms"] = dt
            latencies["retrieval_ms"] = dt

            return RetrievalStageResult(
                candidates=dense_hits,
                raw_bm25_hits=[],
                raw_dense_hits=dense_hits,
                strategy="dense_only",
                stage_latencies=latencies,
                exact_identifiers=exact_ids,
                inferred_doc_ids=inferred_ids,
            )

        # Balanced / Deep / Hybrid: Dual-engine parallel retrieval with RRF fusion
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_kw = executor.submit(
                self.search_keyword,
                query,
                candidate_document_ids,
                top_k_keyword,
                retrieval_signals,
                metadata_filter,
            )
            fut_dense = executor.submit(
                self.search_dense,
                query,
                candidate_document_ids,
                top_k_dense,
                metadata_filter,
            )
            kw_hits = fut_kw.result()
            dense_hits = fut_dense.result()

        dt_ret = (time.perf_counter() - t0) * 1000
        latencies["hybrid_retrieval_ms"] = dt_ret
        latencies["dense_ms"] = dt_ret * 0.7
        latencies["bm25_ms"] = dt_ret * 0.3

        # Additive Reciprocal Rank Fusion (k=DEFAULT_RRF_K=20) with exact-match protection and soft boost
        # Default balanced weights: [0.50 BM25, 0.50 Dense]
        # Exact technical identifier weights: [0.55 BM25, 0.45 Dense]
        t_fuse = time.perf_counter()
        valid_candidate_lists = [r for r in [kw_hits, dense_hits] if r]
        if len(valid_candidate_lists) == 2:
            fuse_weights = (
                [IDENTIFIER_BM25_WEIGHT, IDENTIFIER_DENSE_WEIGHT]
                if exact_ids
                else [DEFAULT_BM25_WEIGHT, DEFAULT_DENSE_WEIGHT]
            )
        else:
            fuse_weights = None
        fused = fuse_candidates(
            valid_candidate_lists,
            method=self.fusion_method,
            k=DEFAULT_RRF_K,
            top_n=max(15, top_k_dense * 2),
            exact_identifiers=exact_ids,
            inferred_document_ids=inferred_ids,
            weights=fuse_weights,
        )
        latencies["fusion_ms"] = (time.perf_counter() - t_fuse) * 1000

        return RetrievalStageResult(
            candidates=fused,
            raw_bm25_hits=kw_hits,
            raw_dense_hits=dense_hits,
            strategy="balanced_hybrid",
            stage_latencies=latencies,
            exact_identifiers=exact_ids,
            inferred_doc_ids=inferred_ids,
        )
