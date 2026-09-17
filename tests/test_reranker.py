from __future__ import annotations

import pytest
from src.components.reranker import CrossEncoderReranker, RerankerResult
from src.components.retrieval.fusion import UnifiedCandidate


def test_cross_encoder_reranker_dedup_and_provenance():
    reranker = CrossEncoderReranker(default_top_k=2)

    cands = [
        UnifiedCandidate(
            chunk_id="c1",
            document_id="d1",
            text="Attention mechanisms allow models to focus on relevant input tokens.",
            score=0.8,
            source_locator="page:3",
            page=3,
        ),
        UnifiedCandidate(
            chunk_id="c1",  # duplicate candidate
            document_id="d1",
            text="Attention mechanisms allow models to focus on relevant input tokens.",
            score=0.7,
            source_locator="page:3",
            page=3,
        ),
        UnifiedCandidate(
            chunk_id="c2",
            document_id="d2",
            text="Gradient descent optimizes weights by calculating partial derivatives.",
            score=0.5,
            source_locator="page:10",
            page=10,
        ),
    ]

    results = reranker.rerank("What is attention in neural networks?", cands, top_k=2)

    assert len(results) == 2
    # Candidate 1 is topically aligned with attention
    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1
    assert results[0].source_locator == "page:3"
    assert results[0].page == 3
    assert results[0].rerank_score is not None
