"""
Backward-compatibility adapter for cross-encoder reranker.
Canonical implementation located in src.components.reranking.reranker.
"""
from .reranking.reranker import CrossEncoderReranker, RerankerResult

__all__ = ["CrossEncoderReranker", "RerankerResult"]
