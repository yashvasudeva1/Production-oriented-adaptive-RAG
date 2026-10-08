from __future__ import annotations

import time
import pytest

from src.components.cache import CacheManager, LRUTTLCache
from src.components.chunking.models import Chunk
from src.components.indexing import MockEmbedder, QdrantIndexer, SearchResult
from src.components.retrieval import (
    UnifiedCandidate,
    fuse_candidates,
    reciprocal_rank_fusion,
    score_normalized_fusion,
    weighted_rrf,
)
from src.evaluation.benchmark import compute_ndcg_at_k
from src.evaluation.ablation import ConfigMetrics, calculate_pareto_frontier


def test_lru_ttl_cache_eviction_and_expiry():
    """Verify LRU capacity eviction and TTL expiration."""
    cache = LRUTTLCache(max_size=3, ttl_seconds=1)

    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") == 1

    # Insert 4th item -> "b" should be evicted as least recently used ("a" was accessed)
    cache.set("d", 4)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4

    # Wait for TTL expiry
    time.sleep(1.1)
    assert cache.get("a") is None
    assert cache.get("c") is None
    assert cache.get("d") is None


def test_cache_manager_invalidation_on_mutation():
    """Verify retrieval cache is invalidated when index version is bumped."""
    mgr = CacheManager.get_instance()
    mgr.enabled = True
    v1 = mgr.index_version

    # Set retrieval result
    mgr.set_retrieval(query="test query", filter_repr="none", top_k=5, mode="dense", results=["hit1", "hit2"])
    cached = mgr.get_retrieval(query="test query", filter_repr="none", top_k=5, mode="dense")
    assert cached == ["hit1", "hit2"]

    # Mutation bumps index version
    mgr.bump_index_version()
    assert mgr.index_version > v1

    # Cached retrieval for old version must be invalidated
    invalidated = mgr.get_retrieval(query="test query", filter_repr="none", top_k=5, mode="dense")
    assert invalidated is None


def test_reranker_pair_caching():
    """Verify reranker pair score caching and retrieval."""
    mgr = CacheManager.get_instance()
    mgr.set_rerank_score("mini-lm", "what is attention?", "attention is all you need text snippet", 0.95)

    score = mgr.get_rerank_score("mini-lm", "what is attention?", "attention is all you need text snippet")
    assert score == pytest.approx(0.95, rel=1e-3)

    miss = mgr.get_rerank_score("mini-lm", "different query", "different snippet")
    assert miss is None


def test_alternative_fusion_strategies():
    """Verify weighted_rrf, score_normalized_fusion, and fuse_candidates dispatcher."""
    c1 = SearchResult(chunk_id="c1", document_id="d1", text="text 1", score=0.9, rank=1, retriever_name="dense")
    c2 = SearchResult(chunk_id="c2", document_id="d2", text="text 2", score=0.4, rank=2, retriever_name="dense")
    c3 = SearchResult(chunk_id="c2", document_id="d2", text="text 2", score=12.5, rank=1, retriever_name="bm25")
    c4 = SearchResult(chunk_id="c3", document_id="d3", text="text 3", score=8.1, rank=2, retriever_name="bm25")

    # 1. Standard RRF
    rrf_res = fuse_candidates([[c1, c2], [c3, c4]], method="rrf", top_n=3)
    assert len(rrf_res) <= 3
    # c2 was present in both dense and BM25 -> should have highest combined score
    assert rrf_res[0].chunk_id == "c2"

    # 2. Weighted RRF (give BM25 5x weight)
    weighted_res = weighted_rrf([[c1, c2], [c3, c4]], weights=[0.2, 1.0], top_n=3)
    assert weighted_res[0].chunk_id == "c2"

    # 3. Score-normalized fusion
    norm_res = score_normalized_fusion([[c1, c2], [c3, c4]], weights=[0.5, 0.5], top_n=3)
    assert len(norm_res) > 0
    assert norm_res[0].chunk_id == "c2"

    # 4. fuse_candidates dispatcher
    disp_res = fuse_candidates([[c1, c2], [c3, c4]], method="score_normalized", top_n=3)
    assert disp_res[0].chunk_id == "c2"


def test_ndcg_at_k_calculation():
    """Verify nDCG@K ranking metric mathematics."""
    # Ideal ranking: relevant at top -> nDCG should be 1.0
    perfect = [1.0, 1.0, 0.0, 0.0]
    assert compute_ndcg_at_k(perfect, k=4) == pytest.approx(1.0, rel=1e-3)

    # Suboptimal ranking: hit at rank 2 instead of rank 1
    suboptimal = [0.0, 1.0, 0.0, 0.0]
    ndcg_sub = compute_ndcg_at_k(suboptimal, k=4)
    assert 0.0 < ndcg_sub < 1.0

    # Optimal hit at rank 1 has higher nDCG than hit at rank 2
    optimal = [1.0, 0.0, 0.0, 0.0]
    assert compute_ndcg_at_k(optimal, k=4) > ndcg_sub

    # Empty or zero hits
    assert compute_ndcg_at_k([], k=4) == 0.0
    assert compute_ndcg_at_k([0.0, 0.0], k=4) == 0.0


def test_pareto_frontier_calculation():
    """Verify Pareto-optimal configuration identification."""
    configs = {
        "cheap_fast": ConfigMetrics(
            config_name="Fast",
            description="",
            recall_at_k=0.70,
            mrr=0.65,
            p50_latency_ms=10.0,
            mean_latency_ms=12.0,
        ),
        "slow_high_quality": ConfigMetrics(
            config_name="Deep",
            description="",
            recall_at_k=0.95,
            mrr=0.90,
            p50_latency_ms=100.0,
            mean_latency_ms=110.0,
        ),
        "strictly_dominated": ConfigMetrics(
            config_name="Dominated",
            description="",
            recall_at_k=0.60,  # Lower quality than fast
            mrr=0.50,
            p50_latency_ms=20.0,  # Higher latency than fast
            mean_latency_ms=25.0,
        ),
    }

    calculate_pareto_frontier(configs)

    assert configs["cheap_fast"].is_pareto_optimal is True
    assert configs["slow_high_quality"].is_pareto_optimal is True
    assert configs["strictly_dominated"].is_pareto_optimal is False
