from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from src.components.indexing.embedder import (
    SentenceTransformerEmbedder,
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
