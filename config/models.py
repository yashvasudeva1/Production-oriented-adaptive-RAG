from __future__ import annotations

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Global configuration settings for Nexus loaded via pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    model_name: str = "Qwen/Qwen3-8B"
    embedding_model: str = "thenlper/gte-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    qdrant_url: str = ""
    qdrant_path: str = ""
    qdrant_collection: str = "nexus_chunks"
    qdrant_distance: str = "Dot"
    qdrant_api_key: str = ""

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
    confidence_high_threshold: float = 0.85
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
    llm_model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0
    offline_eval: bool = False

    groq_api_key: str = Field(default="")
    openrouter_api_key: str = Field(default="")
    hf_token: str = Field(default="")
    langsmith_tracing: bool = False
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = "Nexus"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    allowed_origins: str = "*"


_CONFIG_INSTANCE: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = AppConfig()
    return _CONFIG_INSTANCE
