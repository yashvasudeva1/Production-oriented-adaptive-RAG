from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or research query.")
    document_ids: Optional[List[str]] = Field(
        default=None, description="Optional list of document IDs to scope search to."
    )
    mode: Optional[str] = Field(
        default=None, description="Optional execution mode: fast, balanced, or deep."
    )


class SourceReference(BaseModel):
    citation_id: int
    document_id: str
    filename: str
    chunk_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    source_locator: Optional[str] = None
    snippet: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    abstained: bool = False
    abstention_reason: Optional[str] = None
    query_type: str = "general"
    execution_mode: str = "balanced"
    sources: List[SourceReference] = Field(default_factory=list)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    citation_validation: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    document_ids: Optional[List[str]] = None
    strategy: str = "hybrid"  # dense, keyword, hybrid, parent_child, multi_query


class SearchItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    source_locator: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    retriever: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchItem] = Field(default_factory=list)
    latency_ms: float = 0.0


class DocumentItem(BaseModel):
    document_id: str
    filename: str
    sha256: str
    status: str
    chunk_count: int = 0
    size_bytes: int = 0
    indexed_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    directory: Optional[str] = None
    force_reindex: bool = False


class IngestResponse(BaseModel):
    documents_indexed: int
    documents_skipped: int
    chunks_indexed: int
    duration_ms: float
    errors: List[Dict[str, str]] = Field(default_factory=list)


class IngestJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    qdrant_healthy: bool
    document_count: int
    chunk_count: int
