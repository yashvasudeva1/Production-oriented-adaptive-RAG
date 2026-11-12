from __future__ import annotations

import os
from typing import Any, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_OFFLINE_ENV = os.getenv("OFFLINE_EVAL", "").lower() in ("1", "true", "yes")


class AppConfig(BaseSettings):
    """Global configuration settings for Nexus loaded via pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=None if _OFFLINE_ENV else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    rag_tier: str = "balanced"
    model_name: str = "Qwen/Qwen3-8B"
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = 1024
    reranker_model: Optional[str] = None
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"
    allow_mock_embedder: bool = False

    qdrant_url: str = ""
    qdrant_path: str = ""
    qdrant_collection: str = "nexus_chunks"
    qdrant_distance: str = "Dot"
    qdrant_api_key: str = ""
    qdrant_sparse_enabled: bool = False

    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_full_scan_threshold: int = 10000
    hnsw_on_disk: bool = False
    hnsw_ef_search: Optional[int] = 64

    bm25_index_path: str = "metadata/chunks.json"
    chunk_size: int = 200
    chunk_overlap: int = 30
    top_k: int = 10
    dense_top_k: int = 10
    keyword_top_k: int = 10
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    final_top_k: int = 5

    reranker_enabled: bool = True
    conditional_reranking_enabled: bool = True
    confidence_high_threshold: float = 0.74
    confidence_medium_threshold: float = 0.60
    max_parent_expansions: int = 3
    max_parent_tokens: int = 1200
    max_context_tokens: int = 2048
    metadata_filter_enabled: bool = True
    payload_indexes_enabled: bool = True

    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    fusion_method: str = "rrf"

    ocr_enabled: bool = True
    tesseract_cmd: str = "/usr/bin/tesseract"
    transcribe_audio_video: bool = False
    whisper_model: str = "base"

    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_fallback_provider: str = "offline"
    llm_fallback_model: str = ""
    llm_timeout: float = 10.0
    llm_max_retries: int = 2
    temperature: float = 0.0
    offline_eval: bool = False

    groq_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    openrouter_api_key: str = Field(default="")
    hf_token: str = Field(default="")
    langsmith_tracing: bool = False
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = "Nexus"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    allowed_origins: str = "*"

    @field_validator("embedding_dimension", "hnsw_ef_search", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: Any) -> Any:
        if value == "":
            return None
        return value

    def model_post_init(self, __context: Any) -> None:
        from src.components.orchestration.model_tiers import MODEL_TIERS, resolve_tier_name

        canonical_tier = resolve_tier_name(self.rag_tier)
        if canonical_tier in MODEL_TIERS:
            spec = MODEL_TIERS[canonical_tier]
            if not self.embedding_model:
                self.embedding_model = spec.default_embedding
            if not self.reranker_model:
                self.reranker_model = spec.default_reranker
        else:
            if not self.embedding_model:
                self.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
            if not self.reranker_model:
                self.reranker_model = "Qwen/Qwen3-Reranker-0.6B"

    def get_tier_spec(self):
        """Retrieve model specification for current active tier."""
        from src.components.orchestration.model_tiers import get_tier_spec

        return get_tier_spec(self.rag_tier)

    def validate(self) -> List[str]:
        """Validate configuration settings and return list of warnings/errors."""
        warnings: List[str] = []
        from src.components.orchestration.model_tiers import MODEL_TIERS, resolve_tier_name

        canonical_tier = resolve_tier_name(self.rag_tier)
        if canonical_tier not in MODEL_TIERS:
            warnings.append(
                f"Invalid RAG_TIER: '{self.rag_tier}'. Must be one of: {list(MODEL_TIERS.keys())} or levels 1, 2, 3."
            )
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
