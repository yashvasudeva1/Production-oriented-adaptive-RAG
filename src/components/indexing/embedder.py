from __future__ import annotations

import hashlib
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    def embed_texts(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Embed a sequence of text strings into normalized vectors."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query string."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass


class MockEmbedder(BaseEmbedder):
    """
    Deterministic pseudo-embedding model for offline testing and fast unit tests.
    Generates reproducible unit-normalized vectors via hash projection.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 4) % 2 == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_texts(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_single(query)


TASK_PREFIXES: Dict[str, Dict[str, str]] = {
    "bge": {
        "query": "Represent this sentence for searching relevant passages: ",
        "document": "",
    },
    "e5": {
        "query": "query: ",
        "document": "passage: ",
    },
}


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Sentence-Transformers embedding provider with batch processing,
    dynamic dimension detection, model-aware task prompts, query caching,
    and observable fallback.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._dimension = 384
        self._fallback = MockEmbedder(dim=384)
        self._query_cache: Dict[str, List[float]] = {}
        self._max_cache_size = 512

    def format_text(self, text: str, is_query: bool = False) -> str:
        """Apply model-specific task prompts to queries or passages."""
        clean = text.strip()
        m_lower = self.model_name.lower()
        for key, prefixes in TASK_PREFIXES.items():
            if key in m_lower:
                prefix = prefixes["query"] if is_query else prefixes["document"]
                if prefix and not clean.startswith(prefix):
                    return f"{prefix}{clean}"
        return clean

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    self._model = SentenceTransformer(
                        self.model_name, device=self.device, local_files_only=True
                    )
                except Exception:
                    self._model = SentenceTransformer(
                        self.model_name, device=self.device
                    )
                if hasattr(self._model, "get_sentence_embedding_dimension"):
                    self._dimension = self._model.get_sentence_embedding_dimension()
                elif hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    probe = self._model.encode(["probe"], show_progress_bar=False)
                    self._dimension = len(probe[0])

                self._fallback = MockEmbedder(dim=self._dimension)
                logger.info(
                    f"Loaded SentenceTransformer: {self.model_name} (dim: {self._dimension}, device: {self.device})"
                )
            except Exception as exc:
                allow_mock = (
                    os.getenv("ALLOW_MOCK_EMBEDDER", "0").lower() in ("1", "true", "yes")
                    or os.getenv("OFFLINE_EVAL", "0").lower() in ("1", "true", "yes")
                )
                if not allow_mock:
                    raise RuntimeError(
                        f"Failed to load SentenceTransformer model '{self.model_name}': {exc}. "
                        "Silent fallback to MockEmbedder is disabled in production to protect index integrity. "
                        "Set ALLOW_MOCK_EMBEDDER=1 for offline unit tests."
                    )
                logger.warning(
                    f"Could not load SentenceTransformer '{self.model_name}': {exc}. "
                    "Falling back to deterministic mock embedder (ALLOW_MOCK_EMBEDDER enabled)."
                )
                self._model = False
                self._fallback = MockEmbedder(dim=self._dimension)

    @property
    def dimension(self) -> int:
        if self._model is None:
            self._load_model()
        return self._dimension

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        is_query: bool = False,
    ) -> List[List[float]]:
        self._load_model()
        if not texts:
            return []

        formatted_texts = [self.format_text(t, is_query=is_query) for t in texts]
        bs = batch_size or self.batch_size

        if self._model and self._model is not False:
            try:
                embeddings = self._model.encode(
                    formatted_texts,
                    batch_size=bs,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                vecs = [e.tolist() for e in embeddings]
                for v in vecs:
                    if len(v) != self._dimension:
                        raise ValueError(
                            f"Model output dimension mismatch: got {len(v)}, expected {self._dimension}"
                        )
                return vecs
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                allow_mock = (
                    os.getenv("ALLOW_MOCK_EMBEDDER", "0").lower() in ("1", "true", "yes")
                    or os.getenv("OFFLINE_EVAL", "0").lower() in ("1", "true", "yes")
                )
                if not allow_mock:
                    raise RuntimeError(f"Embedding inference failed: {exc}")
                logger.error(f"Error generating embeddings: {exc}, using fallback.")

        return self._fallback.embed_texts(formatted_texts, batch_size=bs)

    def embed_query(self, query: str) -> List[float]:
        q_formatted = self.format_text(query, is_query=True)
        from ..cache import CacheManager
        cache = CacheManager.get_instance()
        cached = cache.get_embedding(self.model_name, q_formatted)
        if cached is not None:
            return cached

        if q_formatted in self._query_cache:
            return self._query_cache[q_formatted]

        results = self.embed_texts([query], is_query=True)
        vec = results[0] if results else [0.0] * self.dimension

        if len(self._query_cache) >= self._max_cache_size:
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]

        self._query_cache[q_formatted] = vec
        cache.set_embedding(self.model_name, q_formatted, vec)
        return vec
