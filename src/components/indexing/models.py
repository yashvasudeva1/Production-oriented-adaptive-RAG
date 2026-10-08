from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class DocumentRecord:
    """Represents the lifecycle and metadata state of an ingested document."""
    document_id: str
    filename: str
    sha256: str
    status: str = "NEW"  # NEW, INGESTING, INDEXED, FAILED, OUTDATED, DELETED
    chunk_count: int = 0
    size_bytes: int = 0
    indexed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    """Unified retrieval result from dense, keyword, or hybrid retrieval."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int = 0
    parent_id: Optional[str] = None
    source_locator: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_type: str = "text"
    retriever_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    rrf_score: Optional[float] = None

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

    def copy(self) -> Dict[str, Any]:
        d = self.to_dict()
        if self.rrf_score is not None:
            d["rrf_score"] = self.rrf_score
        return d

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IndexingStats:
    """Summary statistics from an indexing job."""
    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_indexed: int = 0
    duration_ms: float = 0.0
    errors: List[Dict[str, str]] = field(default_factory=list)
