# Production Evaluation Report

- **Evaluation Regime:** Deterministic Offline Fallback (Heuristic Token/Set Math)
- **Evaluated At:** 2026-10-18T06:58:51.240293+00:00
- **LLM Evaluator:** None (offline deterministic formulas)
- **Engine Execution:** `deterministic_fallback`
- **Total Test Cases Evaluated:** `66`

> [!NOTE]
> **Evaluation Regime & Metric Clarification:**
> The metrics in this report were generated via the **Deterministic Offline Heuristic Suite** because remote LLM judge endpoints (Groq) encountered rate limits (HTTP 429).
> - **Context Precision / Recall / Faithfulness:** Calculated using exact set-theoretic overlap and claim verification rules.
> - **Answer Correctness:** Evaluated as **unweighted token-set F1** between response and ground truth. It is **not** identical to official RAGAS `AnswerCorrectness` (which uses an LLM judge for atomic statement classification at 0.75 weight blended with embedding cosine similarity at 0.25 weight). Absolute values are lower (~0.11) due to length dilution between detailed responses and compact references.

## 1. Retrieval Configuration Comparison Table

| Configuration | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Answer Correctness (Token F1) | Abstention Acc | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DENSE** | 0.6450 | 0.3724 | 0.9091 | 0.5182 | 0.1534 | 98.5% | 2954.7 ms |
| **BM25** | 0.6534 | 0.3873 | 1.0000 | 0.5182 | 0.1368 | 90.9% | 7.9 ms |
| **HYBRID** | 0.6309 | 0.3579 | 0.8333 | 0.4992 | 0.1456 | 98.5% | 20.3 ms |
| **HYBRID_RERANK** | 0.6566 | 0.3871 | 0.9520 | 0.5166 | 0.1512 | 98.5% | 110.5 ms |
| **ADAPTIVE** | 0.6282 | 0.3604 | 0.8182 | 0.5030 | 0.1439 | 97.0% | 13.7 ms |

## 2. Statistical Uncertainty (95% Bootstrap Confidence Intervals)

| Configuration | Metric | Mean | 95% CI Lower | 95% CI Upper | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| dense | context_precision | 0.6450 | 0.5411 | 0.7444 | 0.4360 |
| dense | context_recall | 0.3724 | 0.3078 | 0.4407 | 0.2705 |
| dense | faithfulness | 0.9091 | 0.8460 | 0.9621 | 0.2360 |
| dense | answer_correctness | 0.1534 | 0.1232 | 0.1881 | 0.1265 |
| bm25 | context_precision | 0.6534 | 0.5523 | 0.7536 | 0.4239 |
| bm25 | context_recall | 0.3873 | 0.3249 | 0.4526 | 0.2638 |
| bm25 | faithfulness | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| bm25 | answer_correctness | 0.1368 | 0.1089 | 0.1685 | 0.1201 |
| hybrid | context_precision | 0.6309 | 0.5285 | 0.7330 | 0.4473 |
| hybrid | context_recall | 0.3579 | 0.2907 | 0.4264 | 0.2801 |
| hybrid | faithfulness | 0.8333 | 0.7273 | 0.9091 | 0.3755 |
| hybrid | answer_correctness | 0.1456 | 0.1166 | 0.1774 | 0.1213 |
| hybrid_rerank | context_precision | 0.6566 | 0.5521 | 0.7644 | 0.4370 |
| hybrid_rerank | context_recall | 0.3871 | 0.3220 | 0.4535 | 0.2730 |
| hybrid_rerank | faithfulness | 0.9520 | 0.9066 | 0.9849 | 0.1595 |
| hybrid_rerank | answer_correctness | 0.1512 | 0.1221 | 0.1863 | 0.1254 |
| adaptive | context_precision | 0.6282 | 0.5244 | 0.7337 | 0.4564 |
| adaptive | context_recall | 0.3604 | 0.2897 | 0.4294 | 0.2867 |
| adaptive | faithfulness | 0.8182 | 0.7121 | 0.8939 | 0.3887 |
| adaptive | answer_correctness | 0.1439 | 0.1157 | 0.1765 | 0.1207 |

## 3. Per-Category Breakdown (Production Adaptive Pipeline)

| Category | Cases | Context Precision | Context Recall | Faithfulness | Relevancy | Correctness | Abstention | P50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exact` | 6 | 0.9722 | 0.6239 | 1.0000 | 0.5495 | 0.1350 | 100.0% | 9.5 ms |
| `conceptual` | 6 | 0.9722 | 0.6631 | 1.0000 | 0.7014 | 0.2210 | 100.0% | 8.5 ms |
| `procedural` | 6 | 0.5972 | 0.3827 | 1.0000 | 0.4876 | 0.1002 | 100.0% | 8.5 ms |
| `comparison` | 6 | 0.9183 | 0.5001 | 1.0000 | 0.6882 | 0.1045 | 100.0% | 23.8 ms |
| `summarization` | 6 | 0.8195 | 0.3237 | 1.0000 | 0.4935 | 0.1100 | 100.0% | 18.5 ms |
| `multi_part` | 6 | 0.5000 | 0.3612 | 1.0000 | 0.4628 | 0.0862 | 100.0% | 19.4 ms |
| `metadata_filtered` | 6 | 0.7833 | 0.4195 | 0.8333 | 0.4634 | 0.1080 | 83.3% | 7.5 ms |
| `multi_hop` | 6 | 0.4396 | 0.2146 | 1.0000 | 0.5939 | 0.0674 | 100.0% | 16.9 ms |
| `unanswerable` | 6 | 0.0000 | 0.0196 | 0.1667 | 0.3000 | 0.3575 | 83.3% | 14.1 ms |
| `false_premise` | 6 | 0.9083 | 0.4559 | 1.0000 | 0.4802 | 0.1496 | 100.0% | 10.3 ms |
| `adversarial` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.3130 | 0.1432 | 100.0% | 13.9 ms |