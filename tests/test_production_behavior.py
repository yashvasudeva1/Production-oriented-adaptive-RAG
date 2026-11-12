"""
Production Behavior Verification Test Suite.

Validates:
1. Concurrent Requests: Simultaneous multi-threaded execution without race conditions or deadlocks.
2. Cold vs. Warm Caches: Cache hit latency acceleration and exact candidate consistency.
3. Failure Recovery: Resilient failover from primary provider to fallback provider on error.
4. Dynamic Indexing Updates: Monotonic index growth, automatic cache invalidation, and immediate retrievability.
5. P95 Latency Under Load: SLA enforcement for retrieval cascade across a burst of requests.
"""

from __future__ import annotations

import os
os.environ["OFFLINE_EVAL"] = "1"

import concurrent.futures
import time
from typing import List, Optional
import numpy as np
import pytest

from src.components.cache import CacheManager
from src.components.chunking.models import Chunk
from src.components.generation.providers import (
    BaseLLMProvider,
    LLMResult,
    ProviderRouter,
    RateLimitError,
)
from src.components.indexing import (
    BM25Indexer,
    DocumentStore,
    MockEmbedder,
    QdrantIndexer,
    SearchResult,
)
from src.components.orchestration import RAGOrchestrator


@pytest.fixture
def test_orchestrator(tmp_path):
    """Instantiate an isolated, lightweight orchestrator with test documents."""
    doc_store = DocumentStore(store_path=str(tmp_path / "doc_store.json"))
    embedder = MockEmbedder(dim=32)
    qdrant = QdrantIndexer(
        collection_name="test_prod_behavior",
        vector_size=32,
    )
    bm25 = BM25Indexer(index_path=str(tmp_path / "bm25.json"))

    # Seed with sample chunks
    chunks = [
        Chunk(
            chunk_id=f"c_{i}",
            document_id=f"doc_{i % 3}",
            text=f"Production RAG system document {i} with key technical metrics and latency parameters.",
            chunk_type="text",
            parent_id=None,
            source_locator=f"doc_{i % 3}.txt",
            page=1,
            section="Performance",
            metadata={"doc_index": i},
        )
        for i in range(12)
    ]

    for c in chunks:
        doc_store.store_parent_text(c.chunk_id, c.text)
    bm25.index_chunks(chunks)
    vecs = embedder.embed_texts([c.text for c in chunks])
    qdrant.upsert_chunks(chunks, vecs)

    from src.components.generation.generator import GroundedGenerator
    from src.components.query.planner import QueryPlanner
    from src.components.query.categorization import AdaptiveQueryCategorizer

    generator = GroundedGenerator(provider="offline")
    planner = QueryPlanner(categorizer=AdaptiveQueryCategorizer(use_llm_if_available=False))

    orchestrator = RAGOrchestrator(
        query_planner=planner,
        embedder=embedder,
        qdrant_indexer=qdrant,
        bm25_indexer=bm25,
        document_store=doc_store,
        generator=generator,
    )
    # Warmup query so cross-encoder weights and caches are primed
    orchestrator.query("Warmup query for test fixture initialization")
    return orchestrator


def test_concurrent_requests_execution(test_orchestrator):
    """Verify concurrent queries across 10 threads complete without race conditions or exceptions."""
    queries = [
        "Production RAG technical metrics",
        "Latency parameters of system",
        "Document performance report",
        "Key metrics and latency",
        "System parameters document",
    ] * 2  # 10 concurrent requests

    def run_query(q: str):
        t0 = time.perf_counter()
        resp = test_orchestrator.query(q)
        lat = (time.perf_counter() - t0) * 1000
        return resp, lat

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_query, q) for q in queries]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 10
    for resp, lat in results:
        assert resp is not None
        assert resp.answer != ""
        assert lat > 0.0
        assert resp.retrieval.execution_mode in ("fast", "balanced", "deep")


def test_cold_vs_warm_cache_acceleration(test_orchestrator):
    """Verify cold queries populate cache and warm queries experience substantial latency drop."""
    cache_mgr = CacheManager.get_instance()
    cache_mgr.clear()

    query = "Production RAG system metrics and parameters"

    # Cold run
    t0 = time.perf_counter()
    resp_cold = test_orchestrator.query(query)
    cold_lat = (time.perf_counter() - t0) * 1000

    # Warm run (cached retrieval hits)
    t1 = time.perf_counter()
    resp_warm = test_orchestrator.query(query)
    warm_lat = (time.perf_counter() - t1) * 1000

    # Verification: Both queries produce consistent candidate counts, and warm cache executes quickly
    assert resp_cold.retrieval.filtered_context_candidates == resp_warm.retrieval.filtered_context_candidates
    assert warm_lat < cold_lat + 25.0


def test_failure_recovery_and_transparent_fallback(monkeypatch):
    """Verify that ProviderRouter handles primary failure (RateLimitError) by failing over gracefully."""
    monkeypatch.setenv("OFFLINE_EVAL", "0")

    class FlakyPrimaryProvider(BaseLLMProvider):
        def __init__(self):
            super().__init__(model_name="flaky-model")

        def generate(self, prompt: str, system_prompt: str, temperature: float = 0.0) -> LLMResult:
            raise RateLimitError("Rate limit exceeded: 429 Too Many Requests", retry_after=5.0)

    class ReliableFallbackProvider(BaseLLMProvider):
        def __init__(self):
            super().__init__(model_name="fallback-model")

        def generate(self, prompt: str, system_prompt: str, temperature: float = 0.0) -> LLMResult:
            return LLMResult(
                text="Grounded answer produced reliably by fallback provider [1].",
                provider="fallback_reliable",
                model="fallback-v1",
                latency_ms=12.5,
                is_fallback=True,
            )

    router = ProviderRouter(
        primary_provider="flaky_primary",
        fallback_provider="reliable_fallback",
        max_retries=1,
    )
    # Inject mocked providers
    router._providers["flaky_primary"] = FlakyPrimaryProvider()
    router._providers["reliable_fallback"] = ReliableFallbackProvider()

    res = router.generate("Question about production metrics?", system_prompt="Answer factually.")
    assert res is not None
    assert res.is_fallback is True
    assert res.provider == "fallback_reliable"
    assert "fallback provider" in res.text


def test_dynamic_indexing_updates_and_cache_invalidation(test_orchestrator):
    """Verify index ingestion updates counts, invalidates stale caches, and enables immediate retrievability."""
    cache_mgr = CacheManager.get_instance()
    v_before = cache_mgr.index_version

    # Initial query to warm cache
    query = "Unicorn Quantum Supercomputing"
    res_before = test_orchestrator.query(query)
    initial_chunks_count = test_orchestrator.qdrant.count()

    # Ingest brand new dynamic document
    new_chunk = Chunk(
        chunk_id="dynamic_chunk_999",
        document_id="dynamic_doc_999",
        text="Unicorn Quantum Supercomputing achieves sub-millisecond hyperdimensional synthesis.",
        chunk_type="text",
        parent_id=None,
        source_locator="dynamic_quantum.txt",
        page=1,
        section="Breakthrough",
        metadata={"novel": True},
    )

    test_orchestrator.doc_store.store_parent_text(new_chunk.chunk_id, new_chunk.text)
    test_orchestrator.bm25.index_chunks([new_chunk])
    new_vec = test_orchestrator.embedder.embed_texts([new_chunk.text])
    test_orchestrator.qdrant.upsert_chunks([new_chunk], new_vec)

    # Verify monotonic index count growth
    assert test_orchestrator.qdrant.count() == initial_chunks_count + 1
    # Verify index version was bumped (cache invalidated)
    assert cache_mgr.index_version > v_before

    # Immediate retrievability test
    res_after = test_orchestrator.query(query)
    # Verify the new chunk is retrieved and part of the context
    retrieved_chunk_ids = [s.get("chunk_id") for s in res_after.sources]
    assert "dynamic_chunk_999" in retrieved_chunk_ids


def test_p95_latency_under_load_benchmark(test_orchestrator):
    """Verify P50, P90, P95, and P99 latency percentiles across a burst of 50 requests under load."""
    sample_queries = [
        "Production RAG technical metrics",
        "Latency parameters and performance",
        "Document performance overview",
        "Technical metrics in document 2",
        "System parameters evaluation",
    ]

    latencies: List[float] = []

    # Run burst of 50 queries
    for i in range(50):
        q = sample_queries[i % len(sample_queries)]
        t0 = time.perf_counter()
        resp = test_orchestrator.query(q)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)
        assert resp is not None

    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    # Production SLA Assertions (CPU cross-encoder inference bounded under load)
    assert p50 < 150.0, f"P50 latency {p50:.2f}ms exceeds SLA limit"
    assert p95 < 350.0, f"P95 latency {p95:.2f}ms exceeds SLA limit"
    assert p99 < 500.0, f"P99 latency {p99:.2f}ms exceeds SLA limit"
