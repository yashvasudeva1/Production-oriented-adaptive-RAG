"""
Comprehensive Unit Tests for Precise Technical Identifier Detection.

Audits:
1. Ordinary natural-language queries (must NOT trigger exact identifiers)
2. Formulas and snake_case parameters (e.g. p_drop, d_k, layer_norm)
3. CamelCase model and architecture names (e.g. ByteNet, ConvS2S, Word2Vec)
4. Technical uppercase acronyms (e.g. BLEU, HNSW, BM25, BERT, LSTM)
5. Alphanumeric hardware and model IDs (e.g. P100, V100, GPT4)
6. Single-character mathematical variables (e.g. N, h, d, k)
7. Exclusion of common English words and articles (e.g. 'A', 'I', 'the', 'transformer')
8. Verification that dynamic weighting in retrieval fusion is triggered only when justified.
"""

from __future__ import annotations

import pytest
from src.components.query.signals import extract_query_signals, extract_technical_identifiers


class TestIdentifierDetection:
    """Test suite verifying precision and recall of technical identifier extraction."""

    @pytest.mark.parametrize(
        "query,expected_ids,should_trigger_dynamic_weighting",
        [
            # Ordinary Natural Language Queries (Zero false positives)
            ("What is the main advantage of the Transformer architecture?", [], False),
            ("How does multi-head attention compare to single-head attention?", [], False),
            ("What are the instructions for the NITI Aayog internship scheme?", [], False),
            ("What is the minimum qualification required for undergraduate applicants?", [], False),
            ("Can you summarize the primary findings of the paper?", [], False),
            ("How does self-attention handle long range dependencies?", [], False),

            # Formulas & Snake_Case Identifiers
            ("What values were chosen for p_drop and d_k?", ["p_drop", "d_k"], True),
            ("Explain how layer_norm is applied before d_ff", ["layer_norm", "d_ff"], True),
            ("What is the dimension of d_model in the base configuration?", ["d_model"], True),
            ("Compute the inner product of queries and keys scaled by 1 / sqrt(d_k)", ["d_k"], True),

            # CamelCase Model and Library Names
            ("How does the model compare with ByteNet and ConvS2S?", ["ByteNet", "ConvS2S"], True),
            ("What was the Word2Vec embedding dimensionality?", ["Word2Vec"], True),
            ("Compare the speed of FastEmbed against HuggingFaceEmbeddings", ["FastEmbed", "HuggingFaceEmbeddings"], True),

            # Uppercase Technical Acronyms
            ("What was the BLEU score achieved on English to German translation?", ["BLEU"], True),
            ("How does HNSW vector indexing compare to inverted BM25?", ["HNSW", "BM25"], True),
            ("Does the model use standard LSTM or GRU recurrent units?", ["LSTM", "GRU"], True),

            # Alphanumeric Hardware & Model Identifiers
            ("How many P100 GPUs were used for training the base model?", ["P100", "GPU"], True),
            ("What were the batch sizes used on NVIDIA V100 systems?", ["V100"], True),

            # Single-Character Mathematical Variables
            ("What does N represent in the encoder stack?", ["N"], True),
            ("How many attention heads h are employed in the multi-head attention mechanism?", ["h"], True),
            ("In the formula, what is the key dimension when d = 64?", ["d"], True),
            ("If k = 5, what is the value of the parameter?", ["k"], True),

            # English Articles and Pronouns Must NOT be detected as variables
            ("What is a Transformer model?", [], False),
            ("I want to know the eligibility criteria for the internship.", [], False),
        ],
    )
    def test_technical_identifier_classification(
        self, query: str, expected_ids: list[str], should_trigger_dynamic_weighting: bool
    ):
        detected = extract_technical_identifiers(query)
        signals = extract_query_signals(query)

        # Check detected identifiers
        assert set(detected) == set(expected_ids), f"Query '{query}' expected {expected_ids}, got {detected}"
        assert set(signals.exact_identifiers) == set(expected_ids)

        # Check dynamic weighting trigger invariant
        has_exact = bool(signals.exact_identifiers)
        assert has_exact == should_trigger_dynamic_weighting, (
            f"Query '{query}' dynamic weighting trigger expected {should_trigger_dynamic_weighting}, got {has_exact}"
        )
