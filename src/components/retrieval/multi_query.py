from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from .dense import DenseRetriever
from .fusion import UnifiedCandidate, reciprocal_rank_fusion
from ..indexing.models import SearchResult

logger = logging.getLogger(__name__)


class MultiQueryRetrievalResponse(BaseModel):
    """Structured response from multi-query retriever."""
    original_query: str
    generated_queries: List[str] = Field(default_factory=list)
    candidate_chunks: int
    returned_chunks: int
    candidate_document_ids: List[str] = Field(default_factory=list)
    results: List[UnifiedCandidate] = Field(default_factory=list)


class MultiQueryRetriever:
    """
    Generates multiple query perspectives to overcome vocabulary mismatch and
    retrieves/fuses candidate chunks across all formulations using RRF.
    """

    def __init__(
        self,
        base_retriever: Optional[DenseRetriever] = None,
        generator_fn: Optional[Callable[[str, int], List[str]]] = None,
        num_queries: int = 3,
        default_top_k: int = 10,
    ) -> None:
        self.base_retriever = base_retriever or DenseRetriever()
        self.generator_fn = generator_fn
        self.num_queries = num_queries
        self.default_top_k = default_top_k
        self._query_cache: Dict[str, List[str]] = {}

    def _generate_queries_heuristic(self, query: str) -> List[str]:
        """Fast offline rule-based query expansion if no LLM is configured."""
        words = query.strip().split()
        if len(words) <= 3:
            return [
                query,
                f"{query} overview",
                f"{query} details architecture",
            ]
        return [
            query,
            f"What are the main concepts and components of {query}?",
            f"Technical specifications and implementation details for {query}",
        ]

    def generate_queries(self, query: str) -> List[str]:
        if query in self._query_cache:
            return self._query_cache[query]

        if self.generator_fn:
            try:
                queries = self.generator_fn(query, self.num_queries)
                if queries:
                    if query not in queries:
                        queries = [query] + queries
                    result = queries[: self.num_queries + 1]
                    self._query_cache[query] = result
                    return result
            except Exception as exc:
                logger.warning(f"Error in multi-query LLM generator: {exc}, using heuristics.")

        result = self._generate_queries_heuristic(query)
        self._query_cache[query] = result
        return result

    def retrieve(
        self,
        query: str,
        *,
        candidate_document_ids: Optional[Sequence[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> MultiQueryRetrievalResponse:
        k = top_k or self.default_top_k
        formulations = self.generate_queries(query)

        candidate_runs: List[List[SearchResult]] = []
        for q in formulations:
            dense_resp = self.base_retriever.retrieve(
                query=q,
                candidate_document_ids=candidate_document_ids,
                filter_criteria=filter_criteria,
                top_k=k,
            )
            run_items = [
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
                    retriever_name="multi_query_dense",
                    metadata=r.metadata,
                )
                for r in dense_resp.results
            ]
            candidate_runs.append(run_items)

        fused = reciprocal_rank_fusion(candidate_runs, k=60, top_n=k)

        return MultiQueryRetrievalResponse(
            original_query=query,
            generated_queries=formulations,
            candidate_chunks=len(fused),
            returned_chunks=len(fused),
            candidate_document_ids=list(candidate_document_ids or []),
            results=fused,
        )
