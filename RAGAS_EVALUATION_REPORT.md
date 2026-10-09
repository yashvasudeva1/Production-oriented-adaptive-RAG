# Production Evaluation Report

- **Evaluation Regime:** Deterministic Offline Fallback (Heuristic Token/Set Math)
- **Evaluated At:** 2026-10-09T14:56:26.316333+00:00
- **LLM Evaluator:** None (offline deterministic formulas)
- **Engine Execution:** `deterministic_fallback`
- **Total Test Cases Evaluated:** `110`

> [!NOTE]
> **Evaluation Regime & Metric Clarification:**
> The metrics in this report were generated via the **Deterministic Offline Heuristic Suite** because remote LLM judge endpoints (Groq) encountered rate limits (HTTP 429).
> - **Context Precision / Recall / Faithfulness:** Calculated using exact set-theoretic overlap and claim verification rules.
> - **Answer Correctness:** Evaluated as **unweighted token-set F1** between response and ground truth. It is **not** identical to official RAGAS `AnswerCorrectness` (which uses an LLM judge for atomic statement classification at 0.75 weight blended with embedding cosine similarity at 0.25 weight). Absolute values are lower (~0.11) due to length dilution between detailed responses and compact references.

## 1. Retrieval Configuration Comparison Table

| Configuration | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Answer Correctness (Token F1) | Abstention Acc | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DENSE** | 0.5937 | 0.3569 | 0.8561 | 0.5352 | 0.1625 | 97.3% | 1235.8 ms |
| **BM25** | 0.6445 | 0.3925 | 0.9333 | 0.5503 | 0.1561 | 98.2% | 2.7 ms |
| **HYBRID** | 0.6182 | 0.3787 | 0.8000 | 0.5330 | 0.1603 | 98.2% | 12.4 ms |
| **HYBRID_RERANK** | 0.6462 | 0.4037 | 0.9773 | 0.5352 | 0.1625 | 97.3% | 105.2 ms |
| **ADAPTIVE** | 0.6197 | 0.3810 | 0.7909 | 0.5357 | 0.1634 | 97.3% | 3.0 ms |

## 2. Statistical Uncertainty (95% Bootstrap Confidence Intervals)

| Configuration | Metric | Mean | 95% CI Lower | 95% CI Upper | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| dense | context_precision | 0.5937 | 0.5120 | 0.6680 | 0.4367 |
| dense | context_recall | 0.3569 | 0.3084 | 0.4047 | 0.2616 |
| dense | faithfulness | 0.8561 | 0.8121 | 0.9000 | 0.2468 |
| dense | answer_correctness | 0.1625 | 0.1391 | 0.1873 | 0.1258 |
| bm25 | context_precision | 0.6445 | 0.5638 | 0.7144 | 0.4291 |
| bm25 | context_recall | 0.3925 | 0.3408 | 0.4397 | 0.2649 |
| bm25 | faithfulness | 0.9333 | 0.8985 | 0.9636 | 0.1781 |
| bm25 | answer_correctness | 0.1561 | 0.1338 | 0.1814 | 0.1233 |
| hybrid | context_precision | 0.6182 | 0.5374 | 0.6911 | 0.4481 |
| hybrid | context_recall | 0.3787 | 0.3231 | 0.4315 | 0.2989 |
| hybrid | faithfulness | 0.8000 | 0.7273 | 0.8727 | 0.4018 |
| hybrid | answer_correctness | 0.1603 | 0.1366 | 0.1849 | 0.1265 |
| hybrid_rerank | context_precision | 0.6462 | 0.5602 | 0.7232 | 0.4439 |
| hybrid_rerank | context_recall | 0.4037 | 0.3533 | 0.4522 | 0.2652 |
| hybrid_rerank | faithfulness | 0.9773 | 0.9591 | 0.9939 | 0.0970 |
| hybrid_rerank | answer_correctness | 0.1625 | 0.1391 | 0.1873 | 0.1258 |
| adaptive | context_precision | 0.6197 | 0.5388 | 0.6929 | 0.4480 |
| adaptive | context_recall | 0.3810 | 0.3248 | 0.4351 | 0.3002 |
| adaptive | faithfulness | 0.7909 | 0.7182 | 0.8636 | 0.4085 |
| adaptive | answer_correctness | 0.1634 | 0.1398 | 0.1884 | 0.1275 |

## 3. Per-Category Breakdown (Production Adaptive Pipeline)

| Category | Cases | Context Precision | Context Recall | Faithfulness | Relevancy | Correctness | Abstention | P50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exact` | 10 | 0.9750 | 0.6210 | 1.0000 | 0.6055 | 0.1592 | 100.0% | 3.2 ms |
| `conceptual` | 10 | 0.9375 | 0.7091 | 1.0000 | 0.7051 | 0.2513 | 100.0% | 2.8 ms |
| `procedural` | 10 | 0.7756 | 0.4712 | 1.0000 | 0.5831 | 0.1104 | 100.0% | 2.6 ms |
| `comparison` | 10 | 0.7201 | 0.4188 | 1.0000 | 0.6935 | 0.1225 | 100.0% | 3.4 ms |
| `summarization` | 10 | 0.6710 | 0.3670 | 0.9000 | 0.5480 | 0.1100 | 90.0% | 3.2 ms |
| `multi_part` | 10 | 0.6833 | 0.4065 | 0.9000 | 0.5071 | 0.1435 | 90.0% | 3.2 ms |
| `metadata_filtered` | 10 | 0.7743 | 0.4861 | 0.9000 | 0.5403 | 0.1250 | 90.0% | 2.8 ms |
| `multi_hop` | 10 | 0.3463 | 0.2342 | 1.0000 | 0.5883 | 0.0634 | 100.0% | 3.2 ms |
| `unanswerable` | 10 | 0.0000 | 0.0000 | 0.0000 | 0.3117 | 0.4325 | 100.0% | 3.4 ms |
| `false_premise` | 10 | 0.9340 | 0.4771 | 1.0000 | 0.5026 | 0.1327 | 100.0% | 2.8 ms |
| `adversarial` | 10 | 0.0000 | 0.0000 | 0.0000 | 0.3078 | 0.1466 | 100.0% | 2.8 ms |