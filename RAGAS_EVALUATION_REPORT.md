# Production Evaluation Report

- **Evaluation Regime:** Official RAGAS Framework (LLM Judge + Embeddings)
- **Evaluated At:** 2026-10-14T18:50:45.151248+00:00
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
| **DENSE** | 0.6145 | 0.3648 | 0.9364 | 0.5024 | 0.1708 | 91.8% | 1304.8 ms |
| **BM25** | 0.6489 | 0.3914 | 1.0000 | 0.4841 | 0.1374 | 73.6% | 4.2 ms |
| **HYBRID** | 0.6048 | 0.3496 | 0.7364 | 0.5018 | 0.1667 | 91.8% | 9.6 ms |
| **HYBRID_RERANK** | 0.6481 | 0.4003 | 0.9864 | 0.5015 | 0.1693 | 91.8% | 111.4 ms |
| **ADAPTIVE** | 0.5944 | 0.3385 | 0.7182 | 0.5003 | 0.1674 | 90.0% | 6.6 ms |

## 2. Statistical Uncertainty (95% Bootstrap Confidence Intervals)

| Configuration | Metric | Mean | 95% CI Lower | 95% CI Upper | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| dense | context_precision | 0.6145 | 0.5301 | 0.6911 | 0.4445 |
| dense | context_recall | 0.3648 | 0.3148 | 0.4126 | 0.2693 |
| dense | faithfulness | 0.9364 | 0.9000 | 0.9682 | 0.1848 |
| dense | answer_correctness | 0.1708 | 0.1442 | 0.1973 | 0.1436 |
| bm25 | context_precision | 0.6489 | 0.5690 | 0.7195 | 0.4305 |
| bm25 | context_recall | 0.3914 | 0.3403 | 0.4394 | 0.2648 |
| bm25 | faithfulness | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| bm25 | answer_correctness | 0.1374 | 0.1134 | 0.1629 | 0.1330 |
| hybrid | context_precision | 0.6048 | 0.5182 | 0.6840 | 0.4629 |
| hybrid | context_recall | 0.3496 | 0.2912 | 0.4052 | 0.3134 |
| hybrid | faithfulness | 0.7364 | 0.6545 | 0.8182 | 0.4426 |
| hybrid | answer_correctness | 0.1667 | 0.1412 | 0.1934 | 0.1423 |
| hybrid_rerank | context_precision | 0.6481 | 0.5620 | 0.7285 | 0.4487 |
| hybrid_rerank | context_recall | 0.4003 | 0.3486 | 0.4482 | 0.2670 |
| hybrid_rerank | faithfulness | 0.9864 | 0.9682 | 1.0000 | 0.0818 |
| hybrid_rerank | answer_correctness | 0.1693 | 0.1438 | 0.1959 | 0.1436 |
| adaptive | context_precision | 0.5944 | 0.5103 | 0.6772 | 0.4697 |
| adaptive | context_recall | 0.3385 | 0.2815 | 0.3940 | 0.3142 |
| adaptive | faithfulness | 0.7182 | 0.6273 | 0.8000 | 0.4519 |
| adaptive | answer_correctness | 0.1674 | 0.1415 | 0.1945 | 0.1438 |

## 3. Per-Category Breakdown (Production Adaptive Pipeline)

| Category | Cases | Context Precision | Context Recall | Faithfulness | Relevancy | Correctness | Abstention | P50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exact` | 10 | 0.9883 | 0.5904 | 1.0000 | 0.5860 | 0.1762 | 100.0% | 6.6 ms |
| `conceptual` | 10 | 0.9442 | 0.7091 | 1.0000 | 0.6947 | 0.2250 | 100.0% | 8.5 ms |
| `procedural` | 10 | 0.6000 | 0.3750 | 0.8000 | 0.5137 | 0.1231 | 80.0% | 4.8 ms |
| `comparison` | 10 | 0.7201 | 0.3900 | 0.9000 | 0.6285 | 0.1031 | 90.0% | 9.2 ms |
| `summarization` | 10 | 0.6877 | 0.3214 | 0.8000 | 0.4693 | 0.0993 | 80.0% | 6.9 ms |
| `multi_part` | 10 | 0.6833 | 0.3770 | 0.8000 | 0.4541 | 0.1339 | 80.0% | 5.9 ms |
| `metadata_filtered` | 10 | 0.5844 | 0.2933 | 0.6000 | 0.4503 | 0.0889 | 60.0% | 4.5 ms |
| `multi_hop` | 10 | 0.3805 | 0.2035 | 1.0000 | 0.5985 | 0.0676 | 100.0% | 7.1 ms |
| `unanswerable` | 10 | 0.0000 | 0.0000 | 0.0000 | 0.3117 | 0.5061 | 100.0% | 6.2 ms |
| `false_premise` | 10 | 0.9500 | 0.4638 | 1.0000 | 0.4892 | 0.1593 | 100.0% | 8.7 ms |
| `adversarial` | 10 | 0.0000 | 0.0000 | 0.0000 | 0.3078 | 0.1591 | 100.0% | 3.9 ms |