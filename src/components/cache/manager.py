from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LRUTTLCache:
    """Thread-safe LRU cache with Time-To-Live (TTL) expiration."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        self.max_size = max(1, max_size)
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            ts, val = self._cache[key]
            if time.time() - ts > self.ttl_seconds:
                del self._cache[key]
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return val

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
            }


class CacheManager:
    """
    Centralized caching manager supporting:
    - Query embedding caching
    - Retrieval result caching (invalidated on index update)
    - Reranker pair scoring cache
    """

    _instance: Optional[CacheManager] = None
    _singleton_lock = threading.RLock()

    def __init__(
        self,
        enabled: bool = True,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
    ) -> None:
        self.enabled = enabled
        self.index_version = 1
        self._lock = threading.RLock()

        self.embedding_cache = LRUTTLCache(max_size=max_size * 2, ttl_seconds=ttl_seconds)
        self.retrieval_cache = LRUTTLCache(max_size=max_size, ttl_seconds=ttl_seconds)
        self.reranker_cache = LRUTTLCache(max_size=max_size * 4, ttl_seconds=ttl_seconds)

    @classmethod
    def get_instance(cls) -> CacheManager:
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def _hash_str(val: str) -> str:
        return hashlib.sha256(val.encode("utf-8")).hexdigest()

    def _full_content_hash(self, text: str) -> str:
        """Compute full SHA-256 hash for document chunks, preventing 200-char truncation collisions."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_embedding(
        self,
        model_name: str,
        text: str,
        model_version: str = "v1",
        config_version: str = "default",
    ) -> Optional[List[float]]:
        if not self.enabled:
            return None
        text_hash = self._full_content_hash(text)
        key = self._hash_str(f"emb:{model_name}:{model_version}:{config_version}:{text_hash}")
        return self.embedding_cache.get(key)

    def set_embedding(
        self,
        model_name: str,
        text: str,
        embedding: List[float],
        model_version: str = "v1",
        config_version: str = "default",
    ) -> None:
        if not self.enabled:
            return
        text_hash = self._full_content_hash(text)
        key = self._hash_str(f"emb:{model_name}:{model_version}:{config_version}:{text_hash}")
        self.embedding_cache.set(key, embedding)

    def get_retrieval(
        self,
        query: str,
        filter_repr: Any,
        top_k: int,
        mode: str,
        model_info: str = "default",
        config_version: str = "default",
    ) -> Optional[Any]:
        if not self.enabled:
            return None
        query_hash = self._hash_str(query.strip().lower())
        try:
            filter_str = json.dumps(filter_repr, sort_keys=True, default=str)
        except Exception:
            filter_str = str(filter_repr)
        filter_hash = self._hash_str(filter_str)
        cfg_hash = self._hash_str(str(config_version))
        key = self._hash_str(
            f"ret:v{self.index_version}:{mode}:{model_info}:{cfg_hash}:{top_k}:{query_hash}:{filter_hash}"
        )
        return self.retrieval_cache.get(key)

    def set_retrieval(
        self,
        query: str,
        filter_repr: Any,
        top_k: int,
        mode: str,
        results: Any,
        model_info: str = "default",
        config_version: str = "default",
    ) -> None:
        if not self.enabled:
            return
        query_hash = self._hash_str(query.strip().lower())
        try:
            filter_str = json.dumps(filter_repr, sort_keys=True, default=str)
        except Exception:
            filter_str = str(filter_repr)
        filter_hash = self._hash_str(filter_str)
        cfg_hash = self._hash_str(str(config_version))
        key = self._hash_str(
            f"ret:v{self.index_version}:{mode}:{model_info}:{cfg_hash}:{top_k}:{query_hash}:{filter_hash}"
        )
        self.retrieval_cache.set(key, results)

    def get_rerank_score(
        self,
        model_name: str,
        query: str,
        text: str,
        model_version: str = "v1",
        config_version: str = "default",
    ) -> Optional[float]:
        if not self.enabled:
            return None
        # Full content hash of candidate text prevents truncation collisions
        text_hash = self._full_content_hash(text)
        query_hash = self._hash_str(query.strip().lower())
        key = self._hash_str(f"rrk:{model_name}:{model_version}:{config_version}:{query_hash}:{text_hash}")
        return self.reranker_cache.get(key)

    def set_rerank_score(
        self,
        model_name: str,
        query: str,
        text: str,
        score: float,
        model_version: str = "v1",
        config_version: str = "default",
    ) -> None:
        if not self.enabled:
            return
        # Full content hash of candidate text prevents truncation collisions
        text_hash = self._full_content_hash(text)
        query_hash = self._hash_str(query.strip().lower())
        key = self._hash_str(f"rrk:{model_name}:{model_version}:{config_version}:{query_hash}:{text_hash}")
        self.reranker_cache.set(key, float(score))

    def bump_index_version(self) -> int:
        """Invalidate all cached retrieval queries after an index modification."""
        with self._lock:
            self.index_version += 1
            self.retrieval_cache.clear()
            logger.info(f"Bumped index version to {self.index_version}; cleared retrieval cache.")
            return self.index_version

    def clear(self) -> None:
        """Clear all in-memory caches."""
        with self._lock:
            self.embedding_cache.clear()
            self.retrieval_cache.clear()
            self.reranker_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "index_version": self.index_version,
            "embedding": self.embedding_cache.stats(),
            "retrieval": self.retrieval_cache.stats(),
            "reranker": self.reranker_cache.stats(),
        }
