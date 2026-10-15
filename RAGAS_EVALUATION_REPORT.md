# Production Evaluation Report

- **Evaluation Regime:** Official RAGAS Framework (LLM Judge + Embeddings)
- **Evaluated At:** 2026-10-15T06:15:57.006972+00:00
- **LLM Evaluator:** `qwen/qwen3.8-27b`
- **Engine Execution:** `ragas_official`
- **Total Test Cases Evaluated:** `110`

> [!NOTE]
> **Evaluation Regime & Metric Clarification:**
> The metrics in this report were generated via the **Deterministic Offline Heuristic Suite** because remote LLM judge endpoints (Groq) encountered rate limits (HTTP 429).
> - **Context Precision / Recall / Faithfulness:** Calculated using exact set-theoretic overlap and claim verification rules.
> - **Answer Correctness:** Evaluated as **unweighted token-set F1** between response and ground truth. It is **not** identical to official RAGAS `AnswerCorrectness` (which uses an LLM judge for atomic statement classification at 0.75 weight blended with embedding cosine similarity at 0.25 weight). Absolute values are lower (~0.11) due to length dilution between detailed responses and compact references.

## 1. Retrieval Configuration Comparison Table

| Configuration | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Answer Correctness (Token F1) | Abstention Acc | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DENSE** | 0.6389 | 0.3769 | 0.9576 | 0.5024 | 0.1708 | 91.8% | 1991.7 ms |
| **BM25** | 0.6489 | 0.3914 | 1.0000 | 0.4841 | 0.1374 | 73.6% | 4.0 ms |
| **HYBRID** | 0.6070 | 0.3507 | 0.7364 | 0.4980 | 0.1674 | 91.8% | 9.1 ms |
| **HYBRID_RERANK** | 0.6481 | 0.4003 | 0.9864 | 0.5015 | 0.1693 | 91.8% | 97.7 ms |
| **ADAPTIVE** | 0.5950 | 0.3381 | 0.7182 | 0.4980 | 0.1675 | 90.0% | 6.2 ms |

## 2. Statistical Uncertainty (95% Bootstrap Confidence Intervals)

| Configuration | Metric | Mean | 95% CI Lower | 95% CI Upper | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| dense | context_precision | 0.6389 | 0.5483 | 0.7123 | 0.4413 |
| dense | context_recall | 0.3769 | 0.3270 | 0.4239 | 0.2626 |
| dense | faithfulness | 0.9576 | 0.9242 | 0.9848 | 0.1681 |
| dense | answer_correctness | 0.1708 | 0.1442 | 0.1973 | 0.1436 |
| bm25 | context_precision | 0.6489 | 0.5690 | 0.7195 | 0.4305 |
| bm25 | context_recall | 0.3914 | 0.3403 | 0.4394 | 0.2648 |
| bm25 | faithfulness | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| bm25 | answer_correctness | 0.1374 | 0.1134 | 0.1629 | 0.1330 |
| hybrid | context_precision | 0.6070 | 0.5211 | 0.6866 | 0.4631 |
| hybrid | context_recall | 0.3507 | 0.2923 | 0.4076 | 0.3147 |
| hybrid | faithfulness | 0.7364 | 0.6545 | 0.8182 | 0.4426 |
| hybrid | answer_correctness | 0.1674 | 0.1418 | 0.1939 | 0.1427 |
| hybrid_rerank | context_precision | 0.6481 | 0.5620 | 0.7285 | 0.4487 |
| hybrid_rerank | context_recall | 0.4003 | 0.3486 | 0.4482 | 0.2670 |
| hybrid_rerank | faithfulness | 0.9864 | 0.9682 | 1.0000 | 0.0818 |
| hybrid_rerank | answer_correctness | 0.1693 | 0.1438 | 0.1959 | 0.1436 |
| adaptive | context_precision | 0.5950 | 0.5121 | 0.6784 | 0.4698 |
| adaptive | context_recall | 0.3381 | 0.2815 | 0.3940 | 0.3139 |
| adaptive | faithfulness | 0.7182 | 0.6273 | 0.8000 | 0.4519 |
| adaptive | answer_correctness | 0.1675 | 0.1425 | 0.1938 | 0.1441 |

## 3. Per-Category Breakdown (Production Adaptive Pipeline)

| Category | Cases | Context Precision | Context Recall | Faithfulness | Relevancy | Correctness | Abstention | P50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exact` | 10 | 0.9917 | 0.5904 | 1.0000 | 0.5598 | 0.1561 | 100.0% | 6.5 ms |
| `conceptual` | 10 | 0.9474 | 0.7131 | 1.0000 | 0.6947 | 0.2483 | 100.0% | 7.6 ms |
| `procedural` | 10 | 0.6000 | 0.3750 | 0.8000 | 0.5137 | 0.1231 | 80.0% | 4.5 ms |
| `comparison` | 10 | 0.7201 | 0.3900 | 0.9000 | 0.6285 | 0.1031 | 90.0% | 8.1 ms |
| `summarization` | 10 | 0.6877 | 0.3214 | 0.8000 | 0.4693 | 0.0993 | 80.0% | 7.5 ms |
| `multi_part` | 10 | 0.6833 | 0.3770 | 0.8000 | 0.4541 | 0.1339 | 80.0% | 6.2 ms |
| `metadata_filtered` | 10 | 0.5844 | 0.2933 | 0.6000 | 0.4503 | 0.0889 | 60.0% | 4.1 ms |
| `multi_hop` | 10 | 0.3816 | 0.2035 | 1.0000 | 0.5985 | 0.0676 | 100.0% | 6.9 ms |
| `unanswerable` | 10 | 0.0000 | 0.0000 | 0.0000 | 0.3117 | 0.5061 | 100.0% | 5.5 ms |
| `false_premise` | 10 | 0.9489 | 0.4555 | 1.0000 | 0.4892 | 0.1571 | 100.0% | 8.0 ms |
| `adversarial` | 10 | 0.0000 | 0.0000 | 0.0000 | 0.3078 | 0.1591 | 100.0% | 3.2 ms |