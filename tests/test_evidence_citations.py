from __future__ import annotations

import pytest
from src.components.generation.citations import (
    CitationBuilder,
    CitationSource,
    CitationValidator,
)
from src.components.orchestration.context_filter import ContextFilter
from src.components.orchestration.evidence_gate import EvidenceGate
from src.components.reranker import RerankerResult


def test_context_filter_budget_and_duplicates():
    cf = ContextFilter(max_context_tokens=20, similarity_threshold=0.8)

    c1 = RerankerResult(
        chunk_id="c1",
        document_id="d1",
        text="The quick brown fox jumps over the lazy dog.",
        rerank_score=1.0,
    )
    c2 = RerankerResult(
        chunk_id="c2",
        document_id="d1",
        text="The quick brown fox jumps over the lazy dog.",  # duplicate
        rerank_score=0.9,
    )
    c3 = RerankerResult(
        chunk_id="c3",
        document_id="d2",
        text="A completely distinct sentence on astronomy and galaxies.",
        rerank_score=0.8,
    )

    filtered = cf.filter_context([c1, c2, c3])
    # c2 should be eliminated as duplicate
    assert len(filtered) <= 2
    assert c1 in filtered
    assert c2 not in filtered


def test_evidence_gate_supported_and_unsupported():
    gate = EvidenceGate(min_supporting_chunks=1, min_confidence=0.3)

    c1 = RerankerResult(
        chunk_id="c1",
        document_id="d1",
        text="The revenue for fiscal year 2024 was 10 million dollars.",
        rerank_score=2.5,
    )

    # Supported query
    verdict_pass = gate.evaluate("What was the revenue in 2024?", [c1])
    assert verdict_pass.allowed is True
    assert verdict_pass.confidence > 0.4
    assert len(verdict_pass.supporting_chunks) == 1

    # Empty candidates
    verdict_empty = gate.evaluate("What was the revenue?", [])
    assert verdict_empty.allowed is False

    # Premise mismatch / ungrounded query
    verdict_mismatch = gate.evaluate(
        "Explain quantum entanglement and teleportation protocols", [c1]
    )
    assert verdict_mismatch.allowed is False


def test_citation_builder_and_validator():
    builder = CitationBuilder()
    validator = CitationValidator()

    c1 = RerankerResult(
        chunk_id="c1",
        document_id="doc_paper",
        text="Self-attention replaces recurrent neural networks.",
        rerank_score=1.0,
        page=2,
        source_locator="page:2",
        metadata={"filename": "attention.pdf"},
    )
    c2 = RerankerResult(
        chunk_id="c2",
        document_id="doc_paper",
        text="Residual connections prevent vanishing gradients.",
        rerank_score=0.8,
        page=3,
        source_locator="page:3",
        metadata={"filename": "attention.pdf"},
    )

    sources = builder.build_sources([c1, c2])
    assert len(sources) == 2
    assert sources[0].citation_id == 1
    assert sources[0].page == 2
    assert sources[1].citation_id == 2
    assert sources[1].page == 3

    # Valid citations
    answer_valid = "Transformers rely on self-attention [1] and residual connections [2]."
    report_valid = validator.validate(answer_valid, sources)
    assert report_valid.is_valid is True
    assert report_valid.valid_citations == [1, 2]
    assert not report_valid.invalid_citations

    # Hallucinated citation [99]
    answer_invalid = "Transformers use magic [99]."
    report_invalid = validator.validate(answer_invalid, sources)
    assert report_invalid.is_valid is False
    assert 99 in report_invalid.hallucinated_citations
