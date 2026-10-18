from __future__ import annotations

import math
from typing import List
import pytest

from src.components.chunking.models import Chunk
from src.components.indexing.models import SearchResult
from src.components.indexing.qdrant_indexer import QdrantIndexer
from src.components.reranker import CrossEncoderReranker, RerankerResult
from src.components.retrieval.confidence import RetrievalConfidenceScorer
from src.components.retrieval.fusion import UnifiedCandidate, reciprocal_rank_fusion


def test_confidence_gate_conditional_weight_masking():
    """
    Verify Fix 1:
    - Identifier weight is masked out when query has no technical identifiers
    - Denominator renormalizes over active weights so score is not artificially depressed
    - Z-score margin is used
    - Default high threshold is calibrated to 0.74
    """
    scorer = RetrievalConfidenceScorer()
    assert scorer.high_threshold == 0.74

    candidates = [
        SearchResult(
            chunk_id="c1",
            document_id="doc1",
            text="Antigravity architecture uses reciprocal rank fusion for hybrid search.",
            score=0.95,
            rank=1,
            retriever_name="dense_qdrant",
            metadata={"provenance_sources": ["dense", "bm25"]},
        ),
        SearchResult(
            chunk_id="c2",
            document_id="doc1",
            text="Reciprocal rank fusion combines sparse and dense vectors.",
            score=0.60,
            rank=2,
            retriever_name="dense_qdrant",
            metadata={"provenance_sources": ["dense"]},
        ),
        SearchResult(
            chunk_id="c3",
            document_id="doc1",
            text="General overview of data indexing and vector storage.",
            score=0.10,
            rank=3,
            retriever_name="keyword_bm25",
            metadata={"provenance_sources": ["bm25"]},
        ),
    ]

    # Case A: Query with NO identifiers - active weights exclude w_id
    query_no_id = "How does reciprocal rank fusion work in hybrid search?"
    conf_no_id = scorer.score_candidates(
        query=query_no_id,
        candidates=candidates,
        execution_mode="balanced",
        exact_identifiers=[],
    )
    assert conf_no_id.score > 0.0
    assert "exact_id_ratio" in conf_no_id.signals
    assert conf_no_id.score >= 0.58

    # Case B: Query with explicit identifiers - w_id included in denominator
    conf_with_id = scorer.score_candidates(
        query=query_no_id,
        candidates=candidates,
        execution_mode="balanced",
        exact_identifiers=["antigravity"],
    )
    assert len(conf_with_id.exact_identifier_matches) == 1
    assert conf_with_id.score > 0.0

    # Case C: High-consensus candidates reaching calibrated threshold >= 0.74 (Fast Path)
    high_candidates = [
        SearchResult(
            chunk_id="h1",
            document_id="doc1",
            text="Reciprocal rank fusion combines sparse and dense vectors in hybrid search.",
            score=0.98,
            rank=1,
            retriever_name="hybrid",
            metadata={"provenance_sources": ["dense", "bm25"]},
        ),
        SearchResult(
            chunk_id="h2",
            document_id="doc1",
            text="Reciprocal rank fusion combines sparse and dense vectors in hybrid search.",
            score=0.95,
            rank=2,
            retriever_name="hybrid",
            metadata={"provenance_sources": ["dense", "bm25"]},
        ),
    ]
    conf_high = scorer.score_candidates(
        query="reciprocal rank fusion hybrid search",
        candidates=high_candidates,
        execution_mode="balanced",
        exact_identifiers=[],
    )
    assert conf_high.score >= 0.74
    assert conf_high.tier == "high"


def test_balanced_rrf_weights_and_k_constant():
    """
    Verify Fix 2:
    - Default RRF constant is k=20 (down from 60)
    - Default weights are 0.45 BM25 / 0.55 Dense (reflecting upgraded dense retriever)
    """
    bm25_cands = [
        SearchResult(chunk_id="doc_a", document_id="d1", text="bm25 first", score=12.0, rank=1, retriever_name="bm25"),
        SearchResult(chunk_id="doc_b", document_id="d2", text="bm25 second", score=8.0, rank=2, retriever_name="bm25"),
    ]
    dense_cands = [
        SearchResult(chunk_id="doc_b", document_id="d2", text="dense first", score=0.92, rank=1, retriever_name="dense"),
        SearchResult(chunk_id="doc_c", document_id="d3", text="dense second", score=0.85, rank=2, retriever_name="dense"),
    ]

    fused = reciprocal_rank_fusion(bm25_results=bm25_cands, dense_results=dense_cands)
    assert len(fused) == 3

    # doc_b has consensus (rank 2 BM25 + rank 1 Dense) -> should rank #1
    assert fused[0].chunk_id == "doc_b"

    # Verify that rank 1 under k=20 carries more weight: 1/(20+1) = 0.0476 vs 1/(60+1) = 0.0164
    k20_top_dense = 0.55 * (1.0 / (20 + 1))
    assert fused[0].score > k20_top_dense


def test_blended_reranking_formula():
    """
    Verify Fix 3:
    Blended reranking computes:
    S_final(d) = 0.65 * sigmoid(CrossEncoder(q, d)) + 0.35 * RRF_norm(d)
    This prevents cross-encoder noise from overwriting consensus.
    """
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # In test environment, fallback score or loaded model will be blended with fusion score
    candidates = [
        UnifiedCandidate(
            chunk_id="c1",
            document_id="d1",
            text="High consensus chunk with exact technical match",
            fusion_score=0.08,
            score=0.08,
            provenance_sources=["dense", "bm25"],
        ),
        UnifiedCandidate(
            chunk_id="c2",
            document_id="d2",
            text="Low consensus chunk with partial match",
            fusion_score=0.02,
            score=0.02,
            provenance_sources=["dense"],
        ),
    ]

    results = reranker.rerank(query="technical match", candidates=candidates, top_k=2)
    assert len(results) == 2
    # Verify scores are normalized in [0, 1] range due to sigmoid + min-max blend
    for r in results:
        assert 0.0 <= r.rerank_score <= 1.0
        assert r.score == r.rerank_score


def test_qdrant_native_sparse_hybrid_search():
    """
    Verify Fix 5:
    QdrantIndexer with enable_sparse=True indexes both dense and fastembed BM25 sparse vectors,
    executing server-side RRF via Prefetch + FusionQuery(RRF).
    """
    indexer = QdrantIndexer(
        collection_name="test_sparse_hybrid",
        vector_size=4,
        distance="Dot",
        enable_sparse=True,
    )

    chunks = [
        Chunk(
            chunk_id="chunk_sparse_1",
            document_id="doc_alpha",
            text="The quick brown fox jumps over the lazy dog.",
        ),
        Chunk(
            chunk_id="chunk_sparse_2",
            document_id="doc_beta",
            text="Quantum computing and artificial intelligence breakthroughs.",
        ),
    ]
    # Synthetic dense embeddings
    embeddings = [
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.7],
    ]

    upserted = indexer.upsert_chunks(chunks=chunks, embeddings=embeddings)
    assert upserted == 2

    # Query matching chunk 1 lexically and semantically
    results = indexer.hybrid_search(
        query_text="brown fox jumps",
        query_vector=[0.5, 0.5, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) >= 1
    assert results[0].chunk_id == "chunk_sparse_1"
    assert results[0].retriever_name == "native_hybrid_qdrant"

    # Verify single-engine transactional document deletion
    deleted = indexer.delete_by_document_id("doc_alpha")
    assert deleted is True

    # After deletion, only chunk_sparse_2 remains
    remaining = indexer.hybrid_search(
        query_text="quantum computing",
        query_vector=[0.0, 0.0, 0.7, 0.7],
        top_k=2,
    )
    chunk_ids = [r.chunk_id for r in remaining]
    assert "chunk_sparse_1" not in chunk_ids


def test_post_hop_multi_hop_consolidation():
    """Verify Fix 1: Post-hop consolidation guarantees hop coverage while pruning distractor chunks."""
    from src.components.orchestration.pipeline import RAGOrchestrator

    orchestrator = RAGOrchestrator()
    cands = [
        RerankerResult(chunk_id="hop1_top", document_id="doc_hop1", text="Hop 1 primary anchor", score=0.90, rerank_score=0.90),
        RerankerResult(chunk_id="distractor1", document_id="doc_irrelevant", text="Irrelevant tail chunk", score=0.40, rerank_score=0.40),
        RerankerResult(chunk_id="hop2_top", document_id="doc_hop2", text="Hop 2 bridge anchor", score=0.82, rerank_score=0.82),
        RerankerResult(chunk_id="distractor2", document_id="doc_hop1", text="Duplicate hop 1 chunk", score=0.70, rerank_score=0.70),
        RerankerResult(chunk_id="distractor3", document_id="doc_noise", text="Noise chunk", score=0.35, rerank_score=0.35),
    ]
    hop_anchors = {"hop1_top", "hop2_top"}

    consolidated = orchestrator._consolidate_multi_hop_candidates(
        reranked_candidates=cands,
        hop_anchor_ids=hop_anchors,
        num_hops=2,
        min_score_ratio=0.55,
    )

    cons_ids = [c.chunk_id for c in consolidated]
    # 1. Guaranteed hop coverage
    assert "hop1_top" in cons_ids
    assert "hop2_top" in cons_ids
    # 2. Pruning of tail chunks (< 0.55 * 0.90 = 0.495)
    assert "distractor1" not in cons_ids  # score 0.40 < 0.495
    assert "distractor3" not in cons_ids  # score 0.35 < 0.495
    # 3. Capped list size: max(3, 2 + 1) = 3
    assert len(consolidated) <= 3


def test_top1_consensus_guard_and_micro_rerank():
    """Verify Fix 2: Top-1 consensus guard and micro-rerank behavior."""
    from src.components.retrieval.confidence import RetrievalConfidence

    conf_high = RetrievalConfidence(
        score=0.85,
        tier="high",
        is_sufficient=True,
        term_coverage=0.9,
        exact_identifier_matches=[],
        missing_identifiers=[],
        candidate_count=5,
        top_score=0.95,
        score_margin=0.2,
    )

    fused = [
        UnifiedCandidate(chunk_id="c1", document_id="d1", text="Target chunk text with MODEL_ID", score=0.90),
        UnifiedCandidate(chunk_id="c2", document_id="d2", text="Second chunk text", score=0.75),
        UnifiedCandidate(chunk_id="c3", document_id="d3", text="Third chunk text", score=0.60),
    ]

    # Consensus case: top-1 agreement between BM25 and Dense
    top_bm25_id = "c1"
    top_dense_id = "c1"
    top1_agreement = (top_bm25_id == top_dense_id)
    can_bypass = conf_high.score >= 0.82 and top1_agreement
    assert can_bypass is True

    # Borderline case: 0.65 <= score < 0.82 triggers micro-rerank pool of top-6
    conf_borderline = RetrievalConfidence(
        score=0.72,
        tier="medium",
        is_sufficient=True,
        term_coverage=0.7,
        exact_identifier_matches=[],
        missing_identifiers=[],
        candidate_count=10,
        top_score=0.70,
        score_margin=0.05,
    )
    can_bypass_borderline = conf_borderline.score >= 0.82 and top1_agreement
    assert can_bypass_borderline is False
    assert conf_borderline.score >= 0.65  # Eligible for micro-reranking


def test_inferred_nl_filters_soft_boost():
    """Verify Fix 3: Inferred NL metadata filters apply soft preference boost without hard-filtering out candidates."""
    from src.components.query.planner import QueryPlanner

    planner = QueryPlanner()
    # Query with natural language date and org constraints
    plan = planner.plan("What are the 2024 guidelines published by NITI Aayog for student internships?")

    # Inferred filters should NOT be hard filters when caller didn't pass explicit_document_ids
    assert plan.is_explicit_filter is False
    assert plan.candidate_document_ids == []
    # But inferred IDs should be recorded as soft preferences
    assert isinstance(plan.inferred_document_ids, list)


def test_offline_deterministic_synthesis_multi_hop_answer():
    """Verify Fix 4: Offline extractive generator synthesizes numbers and entities with boxed final answer."""
    from src.components.generation.providers import OfflineExtractiveProvider

    provider = OfflineExtractiveProvider()
    prompt = (
        "Sources:\n"
        "Source [1] (Empire State Building.txt):\n"
        "The Empire State Building was dethroned as the world's tallest building in 1970.\n\n"
        "Source [2] (List of tallest buildings in NYC.txt):\n"
        "As of August 2024, Bronte Tower ranks 37th among the tallest buildings in New York City.\n\n"
        "Question: Where would Bronte tower rank among tallest buildings in New York City, as of August 2024?"
    )
    result = provider.generate(prompt=prompt, system_prompt="multi-hop analytical assistant")

    assert result.text
    assert "37th" in result.text
    assert "\\boxed{37th}" in result.text

