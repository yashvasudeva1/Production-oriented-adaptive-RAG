from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Global configuration settings for Production-oriented Adaptive RAG."""
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen3-8B"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_dimension: Optional[int] = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768")) if os.getenv("EMBEDDING_DIMENSION") else None
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
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
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    reranker_device: str = field(default_factory=lambda: os.getenv("RERANKER_DEVICE", "cpu"))
    allow_mock_embedder: bool = field(
        default_factory=lambda: os.getenv("ALLOW_MOCK_EMBEDDER", "0").lower() in ("1", "true", "yes")
    )
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    llm_fallback_provider: str = field(default_factory=lambda: os.getenv("LLM_FALLBACK_PROVIDER", "offline"))
    llm_fallback_model: str = field(default_factory=lambda: os.getenv("LLM_FALLBACK_MODEL", ""))
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "10.0")))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    hnsw_m: int = field(default_factory=lambda: int(os.getenv("HNSW_M", "16")))
    hnsw_ef_construct: int = field(default_factory=lambda: int(os.getenv("HNSW_EF_CONSTRUCT", "100")))
    hnsw_full_scan_threshold: int = field(default_factory=lambda: int(os.getenv("HNSW_FULL_SCAN_THRESHOLD", "10000")))
    hnsw_on_disk: bool = field(
        default_factory=lambda: os.getenv("HNSW_ON_DISK", "false").lower() in ("true", "1", "yes")
    )
    hnsw_ef_search: Optional[int] = field(
        default_factory=lambda: int(os.getenv("HNSW_EF_SEARCH", "64")) if os.getenv("HNSW_EF_SEARCH") else None
    )

    def validate(self) -> List[str]:
        """Validate configuration settings and return list of warnings/errors."""
        warnings: List[str] = []
        if self.llm_provider in ("groq", "auto") and not self.groq_api_key and not os.getenv("OFFLINE_EVAL"):
            warnings.append("LLM_PROVIDER is set to groq but GROQ_API_KEY is not configured.")
        if self.llm_provider == "gemini" and not self.gemini_api_key and not os.getenv("OFFLINE_EVAL"):
            warnings.append("LLM_PROVIDER is set to gemini but GEMINI_API_KEY is not configured.")
        if self.llm_provider == "openrouter" and not self.openrouter_api_key and not os.getenv("OFFLINE_EVAL"):
            warnings.append("LLM_PROVIDER is set to openrouter but OPENROUTER_API_KEY is not configured.")
        if self.top_k <= 0:
            warnings.append(f"Invalid TOP_K: {self.top_k}. Must be > 0.")
        if self.hnsw_m <= 0:
            warnings.append(f"Invalid HNSW_M: {self.hnsw_m}. Must be > 0.")
        if self.hnsw_ef_construct <= 0:
            warnings.append(f"Invalid HNSW_EF_CONSTRUCT: {self.hnsw_ef_construct}. Must be > 0.")
        if self.hnsw_full_scan_threshold < 0:
            warnings.append(f"Invalid HNSW_FULL_SCAN_THRESHOLD: {self.hnsw_full_scan_threshold}. Must be >= 0.")
        if self.hnsw_ef_search is not None and self.hnsw_ef_search <= 0:
            warnings.append(f"Invalid HNSW_EF_SEARCH: {self.hnsw_ef_search}. Must be > 0.")
        if self.embedding_dimension is not None and self.embedding_dimension <= 0:
            warnings.append(f"Invalid EMBEDDING_DIMENSION: {self.embedding_dimension}. Must be > 0.")
        return warnings


_CONFIG_INSTANCE: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = AppConfig()
    return _CONFIG_INSTANCE
