"""
ResearchLens modular components:
- Ingestion: Multi-modal document parsing and normalized block extraction.
- Chunking: Structural, semantic, recursive, and parent-child splitters.
- Indexing: Vector storage (Qdrant), keyword index (BM25), and document lifecycle store.
- Retrieval & Reranking: Dense, BM25, multi-query, parent-child, and cross-encoder reranking.
- Query: Normalization, taxonomy categorization, metadata constraint extraction, and planning.
- Orchestration: Context filtering, evidence gate, and end-to-end RAG orchestrator.
- Generation: Grounded answer generation and citation verification.
"""

from .dense_retrieval import DenseRetriever
from .keyword_matching import BM25Retriever
from .parent_child_retrieval import ParentChildRetriever
from .multi_query_retrieval import MultiQueryRetriever
from .retrieval.hybrid import HybridRetriever
from .reranker import CrossEncoderReranker

__all__ = [
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "ParentChildRetriever",
    "MultiQueryRetriever",
    "CrossEncoderReranker",
]
