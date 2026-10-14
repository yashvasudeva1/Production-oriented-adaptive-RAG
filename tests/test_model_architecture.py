from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from src.components.indexing.embedder import (
    SentenceTransformerEmbedder,
    GeminiEmbedder,
    BaseEmbedder,
    MockEmbedder,
    TASK_PREFIXES,
)
from src.components.indexing.qdrant_indexer import (
    QdrantIndexer,
    IncompatibleCollectionError,
)
from src.components.chunking.models import Chunk
from src.components.reranking.reranker import RerankerResult
from src.components.generation.providers import (
    BaseLLMProvider,
    LLMResult,
    OfflineExtractiveProvider,
    ProviderRouter,
    RateLimitError,
    sanitize_secret,
)
from src.components.generation.generator import GroundedGenerator


def test_sanitize_secret():
    text = "Error with Bearer gsk_1234567890abcdef12345678 and key gsk_abcdef1234567890xyz12"
    sanitized = sanitize_secret(text)
    assert "gsk_1234567890abcdef12345678" not in sanitized
    assert "[REDACTED]" in sanitized


def test_embedder_prompt_prefixing():
    # Test BGE prefixing
    bge_embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-small-en-v1.5")
    q_bge = bge_embedder.format_text("What is quantum computing?", is_query=True)
    assert q_bge.startswith("Represent this sentence for searching relevant passages: ")
    doc_bge = bge_embedder.format_text("Quantum computing uses qubits.", is_query=False)
    assert not doc_bge.startswith("Represent this sentence")

    # Test E5 prefixing
    e5_embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-small-v2")
    q_e5 = e5_embedder.format_text("What is quantum computing?", is_query=True)
    assert q_e5.startswith("query: ")
    doc_e5 = e5_embedder.format_text("Quantum computing uses qubits.", is_query=False)
    assert doc_e5.startswith("passage: ")

    # Test GTE prefixing (no prefix)
    gte_embedder = SentenceTransformerEmbedder(model_name="thenlper/gte-small")
    q_gte = gte_embedder.format_text("What is quantum computing?", is_query=True)
    assert q_gte == "What is quantum computing?"

    # Test EmbeddingGemma-2 prefixing
    gemma_embedder = SentenceTransformerEmbedder(model_name="google/embeddinggemma-2")
    q_gemma = gemma_embedder.format_text("What is quantum computing?", is_query=True)
    assert q_gemma == "task: search result | query: What is quantum computing?"
    doc_gemma = gemma_embedder.format_text("Quantum computing uses qubits.", is_query=False)
    assert doc_gemma == "title: none | text: Quantum computing uses qubits."


def test_embedder_strict_failure_in_production(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_EMBEDDER", "0")
    monkeypatch.setenv("OFFLINE_EVAL", "0")

    embedder = SentenceTransformerEmbedder(model_name="nonexistent/fake-model-xyz-123")
    with pytest.raises(RuntimeError) as exc_info:
        embedder.dimension
    assert "Silent fallback to MockEmbedder is disabled in production" in str(exc_info.value)


def test_qdrant_dimension_mismatch_and_recreate():
    # Create in-memory indexer with 384 dimensions
    indexer_384 = QdrantIndexer(collection_name="test_col_mismatch", vector_size=384)
    assert indexer_384.vector_size == 384

    # Attempt to open same collection with 768 dimensions using same client
    with pytest.raises(IncompatibleCollectionError) as exc_info:
        QdrantIndexer(
            collection_name="test_col_mismatch",
            vector_size=768,
            client=indexer_384.client,
        )
    assert "Incompatible vector spaces cannot be mixed" in str(exc_info.value)

    # Recreate collection to migrate to 768
    indexer_384.vector_size = 768
    indexer_384.recreate_collection()
    assert indexer_384.vector_size == 768

    # Now an indexer with 768 can attach safely
    indexer_768 = QdrantIndexer(
        collection_name="test_col_mismatch",
        vector_size=768,
        client=indexer_384.client,
    )
    assert indexer_768.vector_size == 768


def test_qdrant_upsert_vector_dimension_validation():
    indexer = QdrantIndexer(collection_name="test_upsert_val", vector_size=384)
    chunk = Chunk(
        chunk_id="chunk_test_1",
        document_id="doc_1",
        text="Sample text content for dimension validation test.",
    )
    bad_vec = [0.1] * 128  # Wrong dimension (128 != 384)

    with pytest.raises(ValueError) as exc_info:
        indexer.upsert_chunks([chunk], [bad_vec])
    assert "Vector dimension mismatch" in str(exc_info.value)


def test_provider_router_failover(monkeypatch):
    monkeypatch.setenv("OFFLINE_EVAL", "0")
    # Setup mock failing primary provider
    class FailingProvider(BaseLLMProvider):
        def generate(self, prompt, system_prompt, temperature=0.0):
            raise RateLimitError("HTTP 429 Too Many Requests: Rate limit exceeded", retry_after=30.0)

    router = ProviderRouter(
        primary_provider="failing",
        primary_model="fake-primary",
        fallback_provider="offline",
        fallback_model="extractive-deterministic",
    )
    router._providers["failing"] = FailingProvider(model_name="fake-primary")

    prompt = "Sources: [1] Paris is the capital of France.\nQuestion: What is the capital of France?"
    result = router.generate(prompt=prompt, system_prompt="Answer grounding prompt")

    assert result.is_fallback is True
    assert result.provider == "offline"
    assert "Paris" in result.text


def test_grounded_generator_metadata_and_offline():
    generator = GroundedGenerator(provider="offline")
    chunk = RerankerResult(
        chunk_id="c_meta",
        document_id="doc_meta",
        text="The speed of light in vacuum is approximately 299,792,458 meters per second.",
        rerank_score=2.0,
    )

    out = generator.generate("What is the speed of light?", [chunk])
    assert "answer" in out
    assert "metadata" in out
    assert out["metadata"]["provider"] == "offline"
    assert out["metadata"]["latency_ms"] >= 0.0
    assert len(out["sources"]) == 1
    assert out["validation"]["is_valid"] is True


def test_qdrant_hnsw_indexing_configuration():
    indexer = QdrantIndexer(
        collection_name="test_hnsw_cfg",
        vector_size=128,
        hnsw_m=32,
        hnsw_ef_construct=128,
        hnsw_full_scan_threshold=5000,
    )
    hnsw_cfg = indexer.get_hnsw_config()
    assert hnsw_cfg["m"] == 32
    assert hnsw_cfg["ef_construct"] == 128
    assert hnsw_cfg["full_scan_threshold"] == 5000


def test_qdrant_hnsw_dynamic_update():
    indexer = QdrantIndexer(
        collection_name="test_hnsw_update",
        vector_size=128,
        hnsw_m=16,
        hnsw_ef_construct=100,
    )
    success = indexer.update_hnsw_config(m=64, ef_construct=256)
    assert success is True
    assert indexer.hnsw_m == 64
    assert indexer.hnsw_ef_construct == 256
    hnsw_cfg = indexer.get_hnsw_config()
    assert hnsw_cfg["m"] == 64
    assert hnsw_cfg["ef_construct"] == 256


def test_qdrant_hnsw_search_with_ef():
    indexer = QdrantIndexer(
        collection_name="test_hnsw_search",
        vector_size=4,
        hnsw_m=16,
        hnsw_ef_construct=100,
        hnsw_ef_search=64,
    )
    chunk = Chunk(
        chunk_id="chunk_hnsw_1",
        document_id="doc_hnsw",
        text="HNSW graph traversal enables logarithmic approximate nearest neighbor search.",
    )
    vec = [1.0, 0.0, 0.0, 0.0]
    indexer.upsert_chunks([chunk], [vec])

    # Search using default hnsw_ef_search
    results = indexer.search(vec, top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_hnsw_1"

    # Search with query-time explicit hnsw_ef override
    results_override = indexer.search(vec, top_k=1, hnsw_ef=128)
    assert len(results_override) == 1
    assert results_override[0].chunk_id == "chunk_hnsw_1"


def test_app_config_hnsw_validation():
    from config.models import AppConfig

    valid_cfg = AppConfig(hnsw_m=16, hnsw_ef_construct=100, hnsw_full_scan_threshold=10000, hnsw_ef_search=64)
    warnings = [w for w in valid_cfg.validate() if "HNSW" in w]
    assert len(warnings) == 0

    invalid_cfg = AppConfig(hnsw_m=0, hnsw_ef_construct=-10, hnsw_full_scan_threshold=-1, hnsw_ef_search=0)
    invalid_warnings = [w for w in invalid_cfg.validate() if "HNSW" in w]
    assert len(invalid_warnings) == 4


def test_gemini_embedder_api_key_required(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ALLOW_MOCK_EMBEDDER", "0")
    monkeypatch.setenv("OFFLINE_EVAL", "0")

    with pytest.raises(RuntimeError) as exc_info:
        GeminiEmbedder(api_key="")
    assert "GEMINI_API_KEY environment variable is required" in str(exc_info.value)


def test_gemini_embedder_mocked_calls():
    embedder = GeminiEmbedder(
        model_name="models/gemini-embedding-2",
        api_key="test-key-12345",
        dimension=4,
    )
    assert embedder.dimension == 4

    # Test embed_query mocked
    with patch.object(embedder, "_call_api", return_value={"embedding": {"values": [0.5, 0.5, 0.5, 0.5]}}):
        q_vec = embedder.embed_query("quantum computing")
        assert len(q_vec) == 4
        assert q_vec == [0.5, 0.5, 0.5, 0.5]

    # Test embed_texts mocked
    with patch.object(
        embedder,
        "_call_api",
        return_value={"embeddings": [{"values": [0.1, 0.2, 0.3, 0.4]}, {"values": [0.5, 0.6, 0.7, 0.8]}]},
    ):
        doc_vecs = embedder.embed_texts(["doc1", "doc2"])
        assert len(doc_vecs) == 2
        assert len(doc_vecs[0]) == 4
        assert len(doc_vecs[1]) == 4


def test_system_registry_embedder_routing(monkeypatch):
    from src.components.registry import SystemRegistry

    # Reset registry cache
    with SystemRegistry._lock:
        SystemRegistry._embedder = None
        SystemRegistry._config = None

    # Test Gemini routing
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-xyz")
    gemini_inst = SystemRegistry.get_embedder()
    assert isinstance(gemini_inst, GeminiEmbedder)

    # Reset and test SentenceTransformer routing
    with SystemRegistry._lock:
        SystemRegistry._embedder = None
        SystemRegistry._config = None

    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    st_inst = SystemRegistry.get_embedder()
    assert isinstance(st_inst, SentenceTransformerEmbedder)


def test_app_config_embedding_dimension_validation():
    from config.models import AppConfig

    valid_cfg = AppConfig(embedding_dimension=768)
    assert len([w for w in valid_cfg.validate() if "EMBEDDING_DIMENSION" in w]) == 0

    invalid_cfg = AppConfig(embedding_dimension=-1)
    warnings = [w for w in invalid_cfg.validate() if "EMBEDDING_DIMENSION" in w]
    assert len(warnings) == 1
    assert "Invalid EMBEDDING_DIMENSION" in warnings[0]

