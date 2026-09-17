"""
Backward-compatibility adapter for dense retrieval.
Canonical implementation located in src.components.retrieval.dense.
"""
from .retrieval.dense import DenseResult, DenseRetrievalResponse, DenseRetriever

__all__ = ["DenseResult", "DenseRetrievalResponse", "DenseRetriever"]
