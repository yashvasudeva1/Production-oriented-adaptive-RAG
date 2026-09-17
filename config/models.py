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
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "researchlens_chunks")
    )
    bm25_index_path: str = field(
        default_factory=lambda: os.getenv("BM25_INDEX_PATH", "metadata/chunks.json")
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "400")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "10")))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "5")))
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "2048"))
    )
    tesseract_cmd: str = field(
        default_factory=lambda: os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    )
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen/qwen3.8-27b"))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
