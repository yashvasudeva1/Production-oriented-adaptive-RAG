from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Global configuration settings for ResearchLens."""
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen3-8B"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "thenlper/gte-small")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
    )
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "researchlens_chunks")
    )
    qdrant_distance: str = field(
        default_factory=lambda: os.getenv("QDRANT_DISTANCE", "Dot")
    )
    bm25_index_path: str = field(
        default_factory=lambda: os.getenv("BM25_INDEX_PATH", "metadata/chunks.json")
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "200")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "30")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "10")))
    dense_top_k: int = field(default_factory=lambda: int(os.getenv("DENSE_TOP_K", "10")))
    keyword_top_k: int = field(default_factory=lambda: int(os.getenv("KEYWORD_TOP_K", "10")))
    fusion_top_k: int = field(default_factory=lambda: int(os.getenv("FUSION_TOP_K", "20")))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "5")))
    final_top_k: int = field(default_factory=lambda: int(os.getenv("FINAL_TOP_K", "5")))
    reranker_enabled: bool = field(
        default_factory=lambda: os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    conditional_reranking_enabled: bool = field(
        default_factory=lambda: os.getenv("CONDITIONAL_RERANKING_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    confidence_high_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.85"))
    )
    confidence_medium_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.60"))
    )
    max_parent_expansions: int = field(
        default_factory=lambda: int(os.getenv("MAX_PARENT_EXPANSIONS", "3"))
    )
    max_parent_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_PARENT_TOKENS", "1200"))
    )
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "2048"))
    )
    metadata_filter_enabled: bool = field(
        default_factory=lambda: os.getenv("METADATA_FILTER_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    payload_indexes_enabled: bool = field(
        default_factory=lambda: os.getenv("PAYLOAD_INDEXES_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    )
    cache_max_size: int = field(
        default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE", "1000"))
    )
    fusion_method: str = field(
        default_factory=lambda: os.getenv("FUSION_METHOD", "rrf")
    )
    tesseract_cmd: str = field(
        default_factory=lambda: os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    )
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen/qwen3.8-27b"))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))


_CONFIG_INSTANCE: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = AppConfig()
    return _CONFIG_INSTANCE
