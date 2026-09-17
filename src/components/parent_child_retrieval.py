"""
Backward-compatibility adapter for parent-child retrieval.
Canonical implementation located in src.components.retrieval.parent_child.
"""
from .retrieval.parent_child import (
    ParentChildResult,
    ParentChildRetrievalResponse,
    ParentChildRetriever,
)

__all__ = [
    "ParentChildResult",
    "ParentChildRetrievalResponse",
    "ParentChildRetriever",
]
