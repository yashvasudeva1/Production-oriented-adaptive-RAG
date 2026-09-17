from __future__ import annotations

import pytest
from src.components.query import (
    AdaptiveQueryCategorizer,
    AdaptiveQueryMetadataExtractor,
    QueryPlanner,
    decompose_query,
    normalize_query,
)


def test_normalize_query():
    raw = "  What is   the\t\n Transformer architecture? \x00 "
    norm = normalize_query(raw)
    assert norm == "What is the Transformer architecture?"


def test_query_categorizer_heuristics():
    cat = AdaptiveQueryCategorizer(use_llm_if_available=False)

    q_fact, _ = cat.categorize("What was the revenue in 2024?")
    assert q_fact == "fact"

    q_detail, _ = cat.categorize("Explain the complete onboarding process for employees.")
    assert q_detail == "detailed"

    q_gen, _ = cat.categorize("Give me an overview of the platform architecture.")
    assert q_gen == "general"

    q_code, _ = cat.categorize("Show the def forward implementation in PyTorch.")
    assert q_code == "technical_exact"

    q_comp, _ = cat.categorize("Compare CNN and RNN architectures.")
    assert q_comp == "comparison"


def test_query_decomposition():
    q_comp = "Compare ResNet and EfficientNet architectures"
    subs = decompose_query(q_comp)
    assert len(subs) == 3
    assert any("ResNet" in s for s in subs)
    assert any("EfficientNet" in s for s in subs)

    q_simple = "What is backpropagation?"
    assert decompose_query(q_simple) == [q_simple]


def test_query_planner():
    planner = QueryPlanner()
    plan = planner.plan("Explain how self-attention works in detail.")

    assert plan.query_type == "detailed"
    assert plan.requires_parent_child is True
    assert plan.requires_reranking is True
    assert plan.top_k_dense > 0
    assert plan.top_k_keyword > 0
