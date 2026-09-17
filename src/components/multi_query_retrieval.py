"""
Backward-compatibility adapter for multi-query retrieval.
Canonical implementation located in src.components.retrieval.multi_query.
"""
from .retrieval.multi_query import (
    MultiQueryRetrievalResponse,
    MultiQueryRetriever,
)

__all__ = [
    "MultiQueryRetrievalResponse",
    "MultiQueryRetriever",
]
