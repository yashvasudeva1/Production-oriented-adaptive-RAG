from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.components.chunking.models import Chunk
from src.components.indexing import (
    BM25Indexer,
    DocumentStore,
    MockEmbedder,
    QdrantIndexer,
    SearchResult,
)
from src.components.reranking import CrossEncoderReranker, RerankerResult
from src.components.retrieval import (
    DenseRetriever,
    FilterCondition,
    KeywordRetriever,
    ParentChildRetriever,
    RetrievalConfidence,
    RetrievalConfidenceScorer,
    RetrievalFilter,
    reciprocal_rank_fusion,
)
from src.components.retrieval.fusion import UnifiedCandidate
from src.components.orchestration import RAGOrchestrator


def test_qdrant_payload_indexes_idempotent():
    """Verify Qdrant payload indexes are created idempotently without errors."""
    qdrant = QdrantIndexer(collection_name="test_indexes_col", vector_size=32)
    assert qdrant.health_check() is True

    # Re-run create_payload_indexes to verify idempotency
    indexed = qdrant.create_payload_indexes()
    assert "document_id" in indexed
    assert "metadata.year" in indexed
    assert "metadata.department" in indexed


def test_native_retrieval_filter_conversion():
    """Verify canonical RetrievalFilter converts accurately to native Qdrant filter."""
    rf = RetrievalFilter()
    rf.add_hard_filter("year", 2025, operator="eq")
    rf.add_hard_filter("department", "AI", operator="eq")
    rf.document_ids = ["doc_1", "doc_2"]

    q_filter = rf.to_qdrant_filter()
    assert q_filter is not None
    assert hasattr(q_filter, "must")
    assert len(q_filter.must) == 3  # document_id MatchAny + year MatchValue + department MatchValue


def test_dense_and_bm25_hard_filter_equivalence(tmp_path: Path):
    """
    Verify Dense and BM25 operate on the EXACT SAME hard-filtered universe.
    Both must eliminate non-matching documents and retain matching candidates.
    """
    emb = MockEmbedder(dim=32)
    qdrant = QdrantIndexer(collection_name="test_filter_equiv", vector_size=32)
    chunks_path = tmp_path / "chunks.json"
    bm25 = BM25Indexer(chunks_path=chunks_path)

    # Chunk 1: Department AI, Year 2025
    c1 = Chunk(
        chunk_id="c_ai_2025",
        document_id="doc_ai",
        text="Deep learning transformer models in artificial intelligence.",
        metadata={"department": "AI", "year": 2025, "document_type": "report"},
    )
    # Chunk 2: Department Finance, Year 2025
    c2 = Chunk(
        chunk_id="c_fin_2025",
        document_id="doc_fin",
        text="Financial quarterly revenue model and corporate earnings.",
        metadata={"department": "Finance", "year": 2025, "document_type": "report"},
    )
    # Chunk 3: Department AI, Year 2020
    c3 = Chunk(
        chunk_id="c_ai_2020",
        document_id="doc_old",
        text="Legacy neural networks and rule-based artificial intelligence.",
        metadata={"department": "AI", "year": 2020, "document_type": "paper"},
    )

    chunks = [c1, c2, c3]
    vecs = emb.embed_texts([c.text for c in chunks])
    qdrant.upsert_chunks(chunks, vecs)
    bm25.upsert_chunks(chunks)

    # Filter: department == AI AND year == 2025
    filter_ai_2025 = RetrievalFilter()
    filter_ai_2025.add_hard_filter("department", "AI")
    filter_ai_2025.add_hard_filter("year", 2025)

    dense_ret = DenseRetriever(indexer=qdrant, embedder=emb)
    keyword_ret = KeywordRetriever(indexer=bm25)

    dense_res = dense_ret.retrieve("artificial intelligence", metadata_filter=filter_ai_2025)
    keyword_res = keyword_ret.retrieve("artificial intelligence", metadata_filter=filter_ai_2025)

    # Both must match ONLY c1
    assert len(dense_res.results) == 1
    assert dense_res.results[0].chunk_id == "c_ai_2025"

    assert len(keyword_res.results) == 1
    assert keyword_res.results[0].chunk_id == "c_ai_2025"


def test_impossible_filter_returns_empty(tmp_path: Path):
    """Verify impossible metadata filters return zero hits without errors across both branches."""
    emb = MockEmbedder(dim=32)
    qdrant = QdrantIndexer(collection_name="test_impossible", vector_size=32)
    bm25 = BM25Indexer(chunks_path=tmp_path / "chunks.json")

    c = Chunk(
        chunk_id="c_ex",
        document_id="doc_ex",
        text="Standard document text.",
        metadata={"department": "Legal", "year": 2023},
    )
    qdrant.upsert_chunks([c], emb.embed_texts([c.text]))
    bm25.upsert_chunks([c])

    impossible_filter = RetrievalFilter()
    impossible_filter.add_hard_filter("department", "NonExistentDept")

    dense_ret = DenseRetriever(indexer=qdrant, embedder=emb)
    keyword_ret = KeywordRetriever(indexer=bm25)

    dense_res = dense_ret.retrieve("document", metadata_filter=impossible_filter)
    keyword_res = keyword_ret.retrieve("document", metadata_filter=impossible_filter)

    assert len(dense_res.results) == 0
    assert len(keyword_res.results) == 0


def test_conditional_reranking_confidence_policy():
    """Verify reranker conditionally bypasses when confidence is high and executes when uncertain."""
    reranker = CrossEncoderReranker(confidence_threshold=0.85)

    cand = UnifiedCandidate(
        chunk_id="chk_top",
        document_id="doc_1",
        text="Exact definition of attention mechanism.",
        score=0.95,
        fusion_score=0.95,
    )

    # 1. High confidence (0.92 >= 0.85) -> should bypass rerank
    high_conf = RetrievalConfidence(score=0.92, tier="high", is_sufficient=True)
    should_rerank_high = reranker.should_rerank(
        query_type="fact",
        execution_mode="balanced",
        candidates=[cand, cand],
        confidence=high_conf,
    )
    assert should_rerank_high is False

    # 2. Medium / uncertain confidence (0.70 < 0.85) -> should rerank
    med_conf = RetrievalConfidence(score=0.70, tier="medium", is_sufficient=True)
    should_rerank_med = reranker.should_rerank(
        query_type="general",
        execution_mode="balanced",
        candidates=[cand, cand],
        confidence=med_conf,
    )
    assert should_rerank_med is True


def test_retrieval_confidence_signals():
    """Verify confidence scorer computes measurable agreement, margins, and term coverage."""
    scorer = RetrievalConfidenceScorer(confidence_high_threshold=0.85, confidence_medium_threshold=0.60)

    # Strong candidate with hybrid agreement and exact identifier match
    strong_cand = UnifiedCandidate(
        chunk_id="c_top",
        document_id="doc_1",
        text="The learning rate alpha was set to 0.001 with Adam optimizer.",
        score=0.9,
        provenance_sources=["dense_qdrant", "bm25_lexical"],
    )
    second_cand = UnifiedCandidate(
        chunk_id="c_sec",
        document_id="doc_2",
        text="Training procedure notes.",
        score=0.4,
        provenance_sources=["dense_qdrant"],
    )

    conf = scorer.score_candidates(
        query="What was the learning rate alpha?",
        candidates=[strong_cand, second_cand],
        execution_mode="balanced",
        exact_identifiers=["alpha"],
    )

    assert conf.score > 0.60
    assert "term_coverage" in conf.signals
    assert "agreement_ratio" in conf.signals
    assert "alpha" in conf.exact_identifier_matches
    assert conf.is_sufficient is True


def test_controlled_late_parent_expansion(tmp_path: Path):
    """Verify parent expansion enforces max_parents and token budget late."""
    doc_store = DocumentStore(store_path=tmp_path / "pc_store.json")
    doc_store.store_parent_text("p1", "Long parent text section 1 containing background info.")
    doc_store.store_parent_text("p2", "Long parent text section 2 with extra context.")
    doc_store.store_parent_text("p3", "Long parent text section 3 should not be expanded if limit is 2.")

    pc = ParentChildRetriever(document_store=doc_store)

    cands = [
        RerankerResult(chunk_id="c1", document_id="d1", text="Child 1", rerank_score=0.9, parent_id="p1"),
        RerankerResult(chunk_id="c2", document_id="d2", text="Child 2", rerank_score=0.8, parent_id="p2"),
        RerankerResult(chunk_id="c3", document_id="d3", text="Child 3", rerank_score=0.7, parent_id="p3"),
    ]

    expanded = pc.expand_candidates(cands, max_parents=2, max_parent_tokens=1000)

    assert len(expanded) == 3
    assert "Long parent text section 1" in expanded[0].text
    assert "Long parent text section 2" in expanded[1].text
    # Candidate 3 must NOT be expanded because max_parents=2
    assert expanded[2].text == "Child 3"


def test_latency_telemetry_in_rag_response(tmp_path: Path):
    """Verify RAG orchestrator produces fine-grained latency telemetry."""
    emb = MockEmbedder(dim=32)
    qdrant = QdrantIndexer(collection_name="test_lat_telemetry", vector_size=32)
    bm25 = BM25Indexer(chunks_path=tmp_path / "chunks.json")
    doc_store = DocumentStore(store_path=tmp_path / "doc_store.json")

    chunk = Chunk(
        chunk_id="chk_tel",
        document_id="doc_tel",
        text="Production retrieval-augmented generation systems must be latency aware.",
    )
    qdrant.upsert_chunks([chunk], emb.embed_texts([chunk.text]))
    bm25.upsert_chunks([chunk])

    orchestrator = RAGOrchestrator(
        embedder=emb,
        qdrant_indexer=qdrant,
        bm25_indexer=bm25,
        document_store=doc_store,
    )

    resp = orchestrator.query("What is retrieval augmented generation?", mode="fast")

    assert resp.latency_ms > 0
    assert "fast_retrieval_ms" in resp.retrieval.latency_breakdown_ms
    assert resp.retrieval.confidence_score is not None
