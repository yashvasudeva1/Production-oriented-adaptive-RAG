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


def test_claim_support_validation_independent_of_citations():
    """Verify claim support is evaluated accurately with or without bracketed citations."""
    validator = CitationValidator()

    contexts = [
        "The Transformer encoder consists of 6 identical layers with multi-head attention.",
        "Scaled dot-product attention applies a scaling factor of 1 divided by the square root of d_k.",
    ]

    # Grounded answer with NO citations
    ans_uncited_grounded = "The encoder has 6 identical layers. It uses scaled dot-product attention."
    res = validator.validate_claim_support(ans_uncited_grounded, contexts)
    assert res["is_supported"] is True
    assert res["factual_support_ratio"] >= 0.80
    assert len(res["unsupported_claims"]) == 0

    # Hallucinated answer with NO citations
    ans_uncited_hallucinated = "The encoder was fabricated in 1912 by medieval steam engineers."
    res_hallucinated = validator.validate_claim_support(ans_uncited_hallucinated, contexts)
    assert res_hallucinated["is_supported"] is False
    assert res_hallucinated["factual_support_ratio"] == 0.0
    assert len(res_hallucinated["unsupported_claims"]) == 1


def test_adversarial_claim_verification_categories():
    """
    Adversarial verification suite:
    1. Negation inversion
    2. Semantic / antonym contradiction
    3. Incorrect numbers / quantities
    4. Swapped / hallucinated entities
    5. Plausible-sounding claims sharing vocabulary with evidence
    """
    validator = CitationValidator()

    evidence_pool = [
        "The base model employs 8 parallel attention heads and 6 encoder layers.",
        "The NITI Aayog internship is strictly unpaid and honorary without financial remuneration.",
        "Ashish Vaswani and Noam Shazeer authored the Attention Is All You Need paper at Google Brain in 2017.",
        "Self-attention operates across input sequence tokens in parallel without recurrence or convolutions.",
    ]

    # 1. Negation Inversion: Shares 90% content words with evidence 1, but negates the predicate
    claim_negation = "The base model does not employ 8 parallel attention heads."
    res_neg = validator.validate_claim_support(claim_negation, evidence_pool)
    assert res_neg["is_supported"] is False
    assert res_neg["factual_support_ratio"] == 0.0
    assert res_neg["claim_details"][0]["status"] == "contradicted"
    assert "polarity" in res_neg["claim_details"][0]["failure_reason"].lower() or "negation" in res_neg["claim_details"][0]["failure_reason"].lower()

    # 2. Semantic Contradiction / Antonyms: Uses "paid" and "salary" contradicting "unpaid" and "honorary"
    claim_contradiction = "The NITI Aayog internship is paid with a high monthly salary."
    res_contra = validator.validate_claim_support(claim_contradiction, evidence_pool)
    assert res_contra["is_supported"] is False
    assert res_contra["factual_support_ratio"] == 0.0
    assert res_contra["claim_details"][0]["status"] == "contradicted"

    # 3. Incorrect Numbers: Uses 128 and 24 instead of 8 and 6
    claim_bad_nums = "The base model employs 128 parallel attention heads and 24 encoder layers."
    res_nums = validator.validate_claim_support(claim_bad_nums, evidence_pool)
    assert res_nums["is_supported"] is False
    assert res_nums["factual_support_ratio"] == 0.0
    assert res_nums["claim_details"][0]["status"] == "contradicted"
    assert "numeric" in res_nums["claim_details"][0]["failure_reason"].lower()

    # 4. Swapped Entities: Replaces Vaswani & Shazeer with Einstein & Newton
    claim_swapped_entities = "Albert Einstein and Isaac Newton authored the Attention Is All You Need paper at Google Brain in 2017."
    res_entities = validator.validate_claim_support(claim_swapped_entities, evidence_pool)
    assert res_entities["is_supported"] is False
    assert res_entities["factual_support_ratio"] == 0.0
    assert res_entities["claim_details"][0]["status"] == "ungrounded"
    assert "entity" in res_entities["claim_details"][0]["failure_reason"].lower()

    # 5. Plausible-Sounding Claim Sharing Vocabulary: Borrows sequence tokens, self-attention, but introduces LSTM gates
    claim_false_predicate = "Self-attention operates across input sequence tokens using convolutional recurrent LSTM gates."
    res_pred = validator.validate_claim_support(claim_false_predicate, evidence_pool)
    assert res_pred["is_supported"] is False
    assert res_pred["factual_support_ratio"] == 0.0
    assert res_pred["claim_details"][0]["status"] in ("contradicted", "ungrounded")

