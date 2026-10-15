from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class TierModelSpec:
    """
    Specification for a tier in the Production Adaptive RAG model architecture.
    Enforces open-weight licensing, hardware targets, hybrid backbone retention,
    and seamless fallback.
    """
    level: int
    name: str
    display_name: str
    embedding_models: List[str]
    default_embedding: str
    reranker_models: List[str]
    default_reranker: str
    description: str
    hardware_target: str
    license_info: str
    bm25_enabled: bool = True
    rrf_enabled: bool = True
    rrf_bm25_weight: float = 0.82
    rrf_dense_weight: float = 0.18

    def validate_embedding_model(self, model_name: str) -> bool:
        """Check if an embedding model is registered for this tier."""
        return any(model_name.lower() in m.lower() or m.lower() in model_name.lower() for m in self.embedding_models)

    def validate_reranker_model(self, model_name: str) -> bool:
        """Check if a reranker model is registered for this tier."""
        m_clean = (model_name or "none").strip().lower()
        if m_clean in ("none", "", "disabled"):
            return "none" in self.reranker_models
        return any(m_clean in r.lower() or r.lower() in m_clean for r in self.reranker_models)


MODEL_TIERS: Dict[str, TierModelSpec] = {
    "fast": TierModelSpec(
        level=1,
        name="fast",
        display_name="Fast / High-Confidence Path",
        embedding_models=["Qwen/Qwen3-Embedding-0.6B", "google/embeddinggemma-300m"],
        default_embedding="Qwen/Qwen3-Embedding-0.6B",
        reranker_models=["none", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
        default_reranker="none",
        description="Ultra-low latency path for high-confidence queries and hardware-constrained edge/CPU environments.",
        hardware_target="CPU / Edge / 2GB RAM",
        license_info="Apache 2.0 / Gemma Terms (Free Open-Weights)",
        bm25_enabled=True,
        rrf_enabled=True,
        rrf_bm25_weight=0.82,
        rrf_dense_weight=0.18,
    ),
    "balanced": TierModelSpec(
        level=2,
        name="balanced",
        display_name="Balanced / Production Default",
        embedding_models=["Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-m3"],
        default_embedding="Qwen/Qwen3-Embedding-0.6B",
        reranker_models=["Qwen/Qwen3-Reranker-0.6B", "BAAI/bge-reranker-v2-m3"],
        default_reranker="Qwen/Qwen3-Reranker-0.6B",
        description="Balanced production default offering high retrieval precision with sub-150ms reranking latency.",
        hardware_target="Single GPU / 4GB-8GB VRAM / Multi-core CPU",
        license_info="Apache 2.0 / MIT (Free Open-Weights)",
        bm25_enabled=True,
        rrf_enabled=True,
        rrf_bm25_weight=0.82,
        rrf_dense_weight=0.18,
    ),
    "deep": TierModelSpec(
        level=3,
        name="deep",
        display_name="Deep / High-Quality Path",
        embedding_models=["Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B"],
        default_embedding="Qwen/Qwen3-Embedding-4B",
        reranker_models=["Qwen/Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-8B"],
        default_reranker="Qwen/Qwen3-Reranker-4B",
        description="Maximum quality path for complex reasoning, multi-hop queries, and critical research synthesis.",
        hardware_target="GPU Cluster / 16GB-32GB VRAM",
        license_info="Apache 2.0 (Free Open-Weights)",
        bm25_enabled=True,
        rrf_enabled=True,
        rrf_bm25_weight=0.82,
        rrf_dense_weight=0.18,
    ),
}

TIER_ALIASES: Dict[str, str] = {
    "1": "fast",
    "level1": "fast",
    "level_1": "fast",
    "fast": "fast",
    "high-confidence": "fast",
    "2": "balanced",
    "level2": "balanced",
    "level_2": "balanced",
    "balanced": "balanced",
    "production": "balanced",
    "default": "balanced",
    "3": "deep",
    "level3": "deep",
    "level_3": "deep",
    "deep": "deep",
    "high-quality": "deep",
}


def resolve_tier_name(name_or_level: Union[str, int]) -> str:
    """Normalize input tier name or integer level to standard canonical tier key."""
    key = str(name_or_level).strip().lower()
    return TIER_ALIASES.get(key, key)


def get_tier_spec(tier_name_or_level: Union[str, int] = "balanced") -> TierModelSpec:
    """
    Retrieve model specification for a given tier name or level.
    Raises ValueError if tier is unassisted.
    """
    canonical = resolve_tier_name(tier_name_or_level)
    if canonical not in MODEL_TIERS:
        raise ValueError(
            f"Unknown RAG tier '{tier_name_or_level}'. "
            f"Available canonical tiers: {list(MODEL_TIERS.keys())} or levels 1, 2, 3."
        )
    return MODEL_TIERS[canonical]


def list_all_tiers() -> List[TierModelSpec]:
    """Return all tier specifications ordered by level."""
    return sorted(MODEL_TIERS.values(), key=lambda t: t.level)
