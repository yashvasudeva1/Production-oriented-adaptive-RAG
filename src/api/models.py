from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question or research query.")
    document_ids: Optional[List[str]] = Field(
        default=None, max_length=50, description="Optional list of document IDs to scope search to."
    )
    mode: Optional[Literal["fast", "balanced", "deep", "adaptive"]] = Field(
        default=None, description="Optional execution mode: adaptive, fast, balanced, or deep."
    )

    @field_validator("query")
    @classmethod
    def validate_non_empty_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or whitespace.")
        return cleaned

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v.strip().lower() if isinstance(v, str) else v


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
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    document_ids: Optional[List[str]] = Field(default=None, max_length=50)
    strategy: Literal["hybrid", "dense", "keyword"] = "hybrid"
    metadata_filter: Optional[Dict[str, Any]] = None

    @field_validator("query")
    @classmethod
    def validate_non_empty_search_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or whitespace.")
        return cleaned


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
    version: str = "1.1.0"
    qdrant_healthy: bool
    document_count: int
    chunk_count: int
