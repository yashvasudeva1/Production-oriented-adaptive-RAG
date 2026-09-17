from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from .dense import DenseRetriever
from .fusion import UnifiedCandidate, reciprocal_rank_fusion
from .keyword import KeywordRetriever
from ..indexing.bm25_indexer import BM25Indexer
from ..indexing.models import SearchResult
from ..indexing.qdrant_indexer import QdrantIndexer

logger = logging.getLogger(__name__)


class HybridRetrievalResponse(BaseModel):
    """Structured response from parallel hybrid retriever."""
    query: str
    candidate_chunks: int
    returned_chunks: int
    dense_count: int = 0
    keyword_count: int = 0
    dense_latency_ms: float = 0.0
    keyword_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    candidate_document_ids: List[str] = Field(default_factory=list)
    results: List[UnifiedCandidate] = Field(default_factory=list)


class HybridRetriever:
    """
    Parallel hybrid retriever combining vector dense search (Qdrant)
    and lexical keyword search (BM25) concurrently using a thread pool.
    Fuses results via Reciprocal Rank Fusion (RRF) with full provenance retention.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        default_top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.keyword_retriever = keyword_retriever or KeywordRetriever()
        self.default_top_k = default_top_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        *,
        candidate_document_ids: Optional[Sequence[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        retrieval_signals: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        parallel: bool = True,
    ) -> HybridRetrievalResponse:
        query = str(query or "").strip()
        k = top_k or self.default_top_k
        t_start = time.perf_counter()

        if not query:
            return HybridRetrievalResponse(
                query="",
                candidate_chunks=0,
                returned_chunks=0,
                results=[],
            )

        dense_results: List[SearchResult] = []
        keyword_results: List[SearchResult] = []
        dense_latency = 0.0
        keyword_latency = 0.0

        def _run_dense() -> tuple[List[SearchResult], float]:
            t0 = time.perf_counter()
            resp = self.dense_retriever.retrieve(
                query=query,
                candidate_document_ids=candidate_document_ids,
                filter_criteria=filter_criteria,
                top_k=k,
            )
            lat = (time.perf_counter() - t0) * 1000
            items = [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    text=r.text,
                    score=r.score,
                    rank=r.rank,
                    parent_id=r.parent_id,
                    source_locator=r.source_locator,
                    page=r.page,
                    section=r.section,
                    retriever_name="dense_qdrant",
                    metadata=r.metadata,
                )
                for r in resp.results
            ]
            return items, lat

        def _run_keyword() -> tuple[List[SearchResult], float]:
            t0 = time.perf_counter()
            resp = self.keyword_retriever.retrieve(
                query=query,
                candidate_document_ids=candidate_document_ids,
                retrieval_signals=retrieval_signals,
                top_k=k,
            )
            lat = (time.perf_counter() - t0) * 1000
            for r in resp.results:
                r.retriever_name = "bm25_lexical"
            return resp.results, lat

        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(_run_dense)
                keyword_future = executor.submit(_run_keyword)
                dense_results, dense_latency = dense_future.result()
                keyword_results, keyword_latency = keyword_future.result()
        else:
            dense_results, dense_latency = _run_dense()
            keyword_results, keyword_latency = _run_keyword()

        # Fuse rankings using Reciprocal Rank Fusion
        t_fuse = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [dense_results, keyword_results],
            k=self.rrf_k,
            top_n=k,
        )
        fuse_latency = (time.perf_counter() - t_fuse) * 1000
        total_latency = (time.perf_counter() - t_start) * 1000

        return HybridRetrievalResponse(
            query=query,
            candidate_chunks=len(dense_results) + len(keyword_results),
            returned_chunks=len(fused),
            dense_count=len(dense_results),
            keyword_count=len(keyword_results),
            dense_latency_ms=dense_latency,
            keyword_latency_ms=keyword_latency,
            fusion_latency_ms=fuse_latency,
            total_latency_ms=total_latency,
            candidate_document_ids=list(candidate_document_ids or []),
            results=fused,
        )
