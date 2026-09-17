from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from config.models import AppConfig
from .indexing.bm25_indexer import BM25Indexer
from .indexing.document_store import DocumentStore
from .indexing.embedder import SentenceTransformerEmbedder
from .indexing.qdrant_indexer import QdrantIndexer
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class SystemRegistry:
    """
    Thread-safe registry for model instances, indexers, and document store.
    Ensures expensive models (embeddings, cross-encoder) and database clients
    are initialized once and reused across all requests.
    """

    _lock = threading.RLock()
    _embedder: Optional[SentenceTransformerEmbedder] = None
    _reranker: Optional[CrossEncoderReranker] = None
    _qdrant: Optional[QdrantIndexer] = None
    _bm25: Optional[BM25Indexer] = None
    _doc_store: Optional[DocumentStore] = None
    _config: Optional[AppConfig] = None

    @classmethod
    def get_config(cls) -> AppConfig:
        if cls._config is None:
            with cls._lock:
                if cls._config is None:
                    cls._config = AppConfig()
        return cls._config

    @classmethod
    def get_embedder(cls) -> SentenceTransformerEmbedder:
        if cls._embedder is None:
            with cls._lock:
                if cls._embedder is None:
                    cfg = cls.get_config()
                    logger.info("Initializing shared SentenceTransformerEmbedder")
                    cls._embedder = SentenceTransformerEmbedder(model_name=cfg.embedding_model)
        return cls._embedder

    @classmethod
    def get_reranker(cls) -> CrossEncoderReranker:
        if cls._reranker is None:
            with cls._lock:
                if cls._reranker is None:
                    cfg = cls.get_config()
                    logger.info("Initializing shared CrossEncoderReranker")
                    cls._reranker = CrossEncoderReranker(model_name=cfg.reranker_model)
        return cls._reranker

    @classmethod
    def get_doc_store(cls) -> DocumentStore:
        if cls._doc_store is None:
            with cls._lock:
                if cls._doc_store is None:
                    cls._doc_store = DocumentStore()
        return cls._doc_store

    @classmethod
    def get_qdrant(cls) -> QdrantIndexer:
        if cls._qdrant is None:
            with cls._lock:
                if cls._qdrant is None:
                    cfg = cls.get_config()
                    embedder = cls.get_embedder()
                    cls._qdrant = QdrantIndexer(
                        collection_name=cfg.qdrant_collection,
                        url=cfg.qdrant_url if cfg.qdrant_url else None,
                        vector_size=embedder.dimension,
                    )
        return cls._qdrant

    @classmethod
    def get_bm25(cls) -> BM25Indexer:
        if cls._bm25 is None:
            with cls._lock:
                if cls._bm25 is None:
                    cfg = cls.get_config()
                    cls._bm25 = BM25Indexer(chunks_path=cfg.bm25_index_path)
        return cls._bm25

    @classmethod
    def reset(cls) -> None:
        """Reset registry for testing environments."""
        with cls._lock:
            cls._embedder = None
            cls._reranker = None
            cls._qdrant = None
            cls._bm25 = None
            cls._doc_store = None
            cls._config = None
