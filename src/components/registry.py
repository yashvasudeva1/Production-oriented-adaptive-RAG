from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from config.models import AppConfig
from .indexing.bm25_indexer import BM25Indexer
from .indexing.document_store import DocumentStore
from .indexing.embedder import BaseEmbedder, GeminiEmbedder, SentenceTransformerEmbedder
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
    _embedder: Optional[BaseEmbedder] = None
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
    def get_embedder(cls) -> BaseEmbedder:
        if cls._embedder is None:
            with cls._lock:
                if cls._embedder is None:
                    cfg = cls.get_config()
                    m_lower = cfg.embedding_model.lower()
                    if "gemini" in m_lower and "embeddinggemma" not in m_lower:
                        dim = getattr(cfg, "embedding_dimension", None) or 768
                        logger.info(
                            f"Initializing shared GeminiEmbedder ({cfg.embedding_model}, dim: {dim})"
                        )
                        cls._embedder = GeminiEmbedder(
                            model_name=cfg.embedding_model,
                            api_key=cfg.gemini_api_key or os.getenv("GEMINI_API_KEY", ""),
                            dimension=dim,
                        )
                    else:
                        logger.info(
                            f"Initializing shared SentenceTransformerEmbedder ({cfg.embedding_model} on {cfg.embedding_device})"
                        )
                        cls._embedder = SentenceTransformerEmbedder(
                            model_name=cfg.embedding_model,
                            device=cfg.embedding_device,
                        )
        return cls._embedder

    @classmethod
    def get_reranker(cls) -> CrossEncoderReranker:
        if cls._reranker is None:
            with cls._lock:
                if cls._reranker is None:
                    cfg = cls.get_config()
                    logger.info(
                        f"Initializing shared CrossEncoderReranker ({cfg.reranker_model} on {cfg.reranker_device})"
                    )
                    cls._reranker = CrossEncoderReranker(
                        model_name=cfg.reranker_model,
                        device=cfg.reranker_device,
                    )
        return cls._reranker

    @classmethod
    def get_tier_spec(cls):
        return cls.get_config().get_tier_spec()

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
                        distance=cfg.qdrant_distance,
                        hnsw_m=cfg.hnsw_m,
                        hnsw_ef_construct=cfg.hnsw_ef_construct,
                        hnsw_full_scan_threshold=cfg.hnsw_full_scan_threshold,
                        hnsw_on_disk=cfg.hnsw_on_disk,
                        hnsw_ef_search=cfg.hnsw_ef_search,
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
