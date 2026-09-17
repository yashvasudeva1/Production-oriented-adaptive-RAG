from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

QueryType = Literal[
    "fact",
    "detailed",
    "general",
    "technical_exact",
    "comparison",
    "metadata_constrained",
    "unanswerable",
]

QueryComplexity = Literal["low", "medium", "high"]
ExecutionMode = Literal["fast", "balanced", "deep"]
Answerability = Literal["supported", "unsupported", "unanswerable"]


class QueryPlan(BaseModel):
    """
    Structured execution plan produced by the query planner.
    Acts as the central brain dictating execution mode, candidate scoping,
    retriever routing, candidate budgets, and reranking requirements.
    """
    query: str
    normalized_query: str
    query_type: QueryType = "general"
    complexity: QueryComplexity = "low"
    execution_mode: ExecutionMode = "balanced"
    answerability: Answerability = "supported"

    sub_queries: List[str] = Field(default_factory=list)
    candidate_document_ids: List[str] = Field(default_factory=list)
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    retrieval_signals: Dict[str, Any] = Field(default_factory=dict)

    requires_dense: bool = True
    requires_keyword: bool = True
    requires_dense_search: bool = True
    requires_keyword_search: bool = True
    requires_hybrid: bool = True
    requires_parent_child: bool = False
    requires_multi_query: bool = False
    requires_decomposition: bool = False
    requires_metadata_filter: bool = False
    requires_reranking: bool = True

    top_k: int = 10
    top_k_dense: int = 10
    top_k_keyword: int = 10
    rerank_top_k: int = 5
    context_budget: int = 2048

    confidence: float = 0.90
    reasoning: str = ""
