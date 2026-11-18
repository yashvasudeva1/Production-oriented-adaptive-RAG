from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..indexing.models import SearchResult
from ..reranker import RerankerResult


class RetrievalMetadata(BaseModel):
    """Execution telemetry and candidate provenance metadata."""
    strategy: str
    query_type: str
    execution_mode: str = "balanced"
    cascade_stages_executed: List[str] = Field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None
    reranked: bool = False
    parent_expanded: bool = False
    multi_query_used: bool = False
    decomposed: bool = False
    iterative_hops: int = 1
    bridge_queries: List[str] = Field(default_factory=list)
    fast_rank: Optional[int] = None
    balanced_rank: Optional[int] = None
    deep_rank: Optional[int] = None
    final_rank: Optional[int] = None
    dense_candidates: int = 0
    keyword_candidates: int = 0
    multi_query_candidates: int = 0
    fused_candidates: int = 0
    reranked_candidates: int = 0
    rerank_pool_size: int = 0
    parent_expanded_count: int = 0
    filtered_context_candidates: int = 0
    filter_selectivity: Optional[float] = None
    confidence_score: Optional[float] = None
    confidence_tier: Optional[str] = None
    confidence_signals: Dict[str, float] = Field(default_factory=dict)
    latency_breakdown_ms: Dict[str, float] = Field(default_factory=dict)
    # Pre-generation retrieved candidate records (decoupled from citation validation & abstention)
    retrieved_candidates: List[Dict[str, Any]] = Field(default_factory=list)


class RAGResponse(BaseModel):
    """Complete, structured response from the Nexus RAG platform."""
    query: str
    answer: str
    confidence: float
    abstained: bool = False
    abstention_reason: Optional[str] = None
    query_type: str = "general"
    execution_mode: str = "balanced"
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval: RetrievalMetadata
    citation_validation: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class RetrievalStageResult(BaseModel):
    """Normalized contract returned by the retrieval stage."""
    candidates: List[Any] = Field(default_factory=list)
    raw_bm25_hits: List[SearchResult] = Field(default_factory=list)
    raw_dense_hits: List[SearchResult] = Field(default_factory=list)
    strategy: str = "hybrid"
    stage_latencies: Dict[str, float] = Field(default_factory=dict)
    exact_identifiers: List[str] = Field(default_factory=list)
    inferred_doc_ids: List[str] = Field(default_factory=list)


class RerankStageResult(BaseModel):
    """Normalized contract returned by the conditional reranking stage."""
    candidates: List[RerankerResult] = Field(default_factory=list)
    did_rerank: bool = False
    can_bypass: bool = False
    rerank_pool_size: int = 0
    rerank_latency_ms: float = 0.0
