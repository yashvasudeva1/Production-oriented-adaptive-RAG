"""
Automated Unit and Integration Tests for RAGAS Evaluation Framework.

Validates schema compliance, metric computation math, edge case handling,
regression verification, and dataset splitting.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.evaluation.ragas_dataset import RagasDatasetBuilder, RagasEvalSample
from src.evaluation.ragas_metrics import (
    DeterministicEvaluatorEngine,
    METRIC_REGISTRY,
    StatisticalAnalyzer,
)
from src.evaluation.ragas_eval import RagasEvaluator, RagasReportExporter, RegressionChecker


def test_ragas_dataset_schema_validation():
    """Verify that RagasDatasetBuilder correctly validates required schemas."""
    builder = RagasDatasetBuilder()

    valid_sample = {
        "user_input": "What is the Transformer architecture?",
        "retrieved_contexts": ["The Transformer is based entirely on self-attention mechanisms."],
        "response": "The Transformer uses self-attention without recurrence.",
        "reference": "The Transformer is a sequence transduction model based on attention.",
    }
    is_valid, errors = builder.validate_sample_schema(valid_sample)
    assert is_valid is True
    assert len(errors) == 0

    # Missing field
    invalid_sample = {
        "user_input": "Question?",
        "retrieved_contexts": ["Context"],
        # Missing response and reference
    }
    is_valid, errors = builder.validate_sample_schema(invalid_sample)
    assert is_valid is False
    assert any("response" in e for e in errors)
    assert any("reference" in e for e in errors)


def test_empty_and_malformed_context_handling():
    """Ensure metric calculations handle empty or missing context gracefully without crashing."""
    prec = DeterministicEvaluatorEngine.evaluate_context_precision(
        query="test query", retrieved_contexts=[], reference="test reference"
    )
    assert prec == 0.0

    rec = DeterministicEvaluatorEngine.evaluate_context_recall(
        query="test query", retrieved_contexts=[], reference="test reference"
    )
    assert rec == 0.0

    faith = DeterministicEvaluatorEngine.evaluate_faithfulness(
        response="This is a test statement.", retrieved_contexts=[]
    )
    assert faith == 0.0

    corr = DeterministicEvaluatorEngine.evaluate_answer_correctness(
        response="", reference="Some ground truth."
    )
    assert corr == 0.0


def test_faithfulness_and_hallucination_detection():
    """Verify faithfulness detects grounded vs ungrounded assertions."""
    context = ["The base Transformer model has 6 encoder layers and 8 attention heads."]

    # Grounded response -> High faithfulness
    grounded_resp = "The Transformer model has 6 encoder layers and 8 attention heads."
    f_high = DeterministicEvaluatorEngine.evaluate_faithfulness(grounded_resp, [context[0]])
    assert f_high >= 0.8

    # Hallucinated response -> Low faithfulness
    hallucinated_resp = "The Transformer was invented in 1850 by Thomas Edison using steam engines."
    f_low = DeterministicEvaluatorEngine.evaluate_faithfulness(hallucinated_resp, [context[0]])
    assert f_low <= 0.2


def test_unanswerable_abstention_handling():
    """Verify correct abstention is treated as faithful and non-hallucinated."""
    abstain_resp = "I cannot provide a grounded answer based on the available sources: No sufficient context."
    contexts = ["General guidelines about something unrelated."]

    faith = DeterministicEvaluatorEngine.evaluate_faithfulness(abstain_resp, contexts)
    assert faith == 1.0


def test_stratified_dataset_splitting():
    """Verify reproducible train/dev/test dataset partitioning."""
    builder = RagasDatasetBuilder()
    test_cases = builder.get_split(split="test", train_ratio=0.2, dev_ratio=0.2, seed=42)
    dev_cases = builder.get_split(split="dev", train_ratio=0.2, dev_ratio=0.2, seed=42)
    train_cases = builder.get_split(split="train", train_ratio=0.2, dev_ratio=0.2, seed=42)

    total = len(test_cases) + len(dev_cases) + len(train_cases)
    assert total == len(builder.cases)
    # Ensure no ID overlap across splits
    test_ids = {c.query_id for c in test_cases}
    dev_ids = {c.query_id for c in dev_cases}
    train_ids = {c.query_id for c in train_cases}
    assert len(test_ids & dev_ids) == 0
    assert len(test_ids & train_ids) == 0


def test_statistical_analyzer_bootstrap():
    """Verify bootstrap confidence interval computation."""
    scores = [0.8, 0.85, 0.9, 0.95, 0.75, 0.88]
    stats = StatisticalAnalyzer.compute_stats(scores, n_bootstrap=500)
    assert stats["count"] == 6
    assert 0.7 <= stats["mean"] <= 0.95
    assert stats["ci_95_low"] <= stats["mean"] <= stats["ci_95_high"]


def test_regression_checker_detection(tmp_path: Path):
    """Verify regression checker flags drops larger than allowed threshold."""
    baseline = {
        "configurations": {
            "adaptive": {
                "summary_metrics": {
                    "context_precision": {"mean": 0.85},
                    "context_recall": {"mean": 0.80},
                    "faithfulness": {"mean": 0.90},
                    "answer_correctness": {"mean": 0.75},
                }
            }
        }
    }
    b_file = tmp_path / "baseline.json"
    with open(b_file, "w") as f:
        json.dump(baseline, f)

    # Identical or higher scores -> Pass
    current_pass = {
        "configurations": {
            "adaptive": {
                "summary_metrics": {
                    "context_precision": {"mean": 0.86},
                    "context_recall": {"mean": 0.81},
                    "faithfulness": {"mean": 0.92},
                    "answer_correctness": {"mean": 0.76},
                }
            }
        }
    }
    passed, issues = RegressionChecker.check_regression(current_pass, str(b_file), max_allowed_drop=0.03)
    assert passed is True
    assert len(issues) == 0

    # Severe drop -> Fail
    current_regressed = {
        "configurations": {
            "adaptive": {
                "summary_metrics": {
                    "context_precision": {"mean": 0.70},  # 0.15 drop
                    "context_recall": {"mean": 0.80},
                    "faithfulness": {"mean": 0.90},
                    "answer_correctness": {"mean": 0.75},
                }
            }
        }
    }
    passed, issues = RegressionChecker.check_regression(current_regressed, str(b_file), max_allowed_drop=0.03)
    assert passed is False
    assert len(issues) > 0
    assert "context_precision" in issues[0]


def test_ragas_evaluation_smoke_run():
    """Verify that RagasEvaluator executes successfully over test queries."""
    from unittest.mock import MagicMock
    from src.components.orchestration.pipeline import RAGResponse, RetrievalMetadata

    mock_orch = MagicMock()
    mock_orch.qdrant.count.return_value = 10
    mock_orch.bm25.count.return_value = 10
    mock_orch.query.return_value = RAGResponse(
        query="What is the dimension d_model of the Transformer base model?",
        answer="In the base Transformer model, the embedding and hidden representation dimension d_model is 512 [1].",
        confidence=0.92,
        execution_mode="balanced",
        sources=[{"chunk_id": "c_1", "text": "In the base Transformer model, the embedding dimension d_model is 512.", "score": 0.85}],
        abstained=False,
        retrieval=RetrievalMetadata(strategy="hybrid", query_type="general"),
    )

    evaluator = RagasEvaluator(orchestrator=mock_orch, engine_type="deterministic")
    cases = evaluator.dataset_builder.cases[:3]
    res = evaluator.evaluate_configuration(cases, config_name="adaptive")

    assert res["total_samples"] == 3
    assert "summary_metrics" in res
    assert "context_precision" in res["summary_metrics"]
    assert "faithfulness" in res["summary_metrics"]
    assert len(res["records"]) == 3
