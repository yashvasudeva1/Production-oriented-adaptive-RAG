# Production Evaluation Report

- **Evaluation Regime:** Deterministic Offline Fallback (Heuristic Token/Set Math)
- **Evaluated At:** 2026-10-18T07:50:09.609298+00:00
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
| **DENSE** | 0.6450 | 0.3724 | 0.8182 | 0.5563 | 0.1610 | 97.0% | 3165.5 ms |
| **BM25** | 0.6534 | 0.3873 | 1.0000 | 0.5347 | 0.1435 | 97.0% | 7.9 ms |
| **HYBRID** | 0.6389 | 0.3565 | 0.7836 | 0.5360 | 0.1468 | 98.5% | 15.2 ms |
| **HYBRID_RERANK** | 0.6553 | 0.3898 | 0.8540 | 0.5563 | 0.1610 | 97.0% | 13.9 ms |
| **ADAPTIVE** | 0.6397 | 0.3557 | 0.7523 | 0.5454 | 0.1556 | 97.0% | 9.1 ms |

## 2. Statistical Uncertainty (95% Bootstrap Confidence Intervals)

| Configuration | Metric | Mean | 95% CI Lower | 95% CI Upper | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| dense | context_precision | 0.6450 | 0.5411 | 0.7444 | 0.4360 |
| dense | context_recall | 0.3724 | 0.3078 | 0.4407 | 0.2705 |
| dense | faithfulness | 0.8182 | 0.7487 | 0.8788 | 0.2679 |
| dense | answer_correctness | 0.1610 | 0.1341 | 0.1923 | 0.1188 |
| bm25 | context_precision | 0.6534 | 0.5523 | 0.7536 | 0.4239 |
| bm25 | context_recall | 0.3873 | 0.3249 | 0.4526 | 0.2638 |
| bm25 | faithfulness | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| bm25 | answer_correctness | 0.1435 | 0.1171 | 0.1750 | 0.1160 |
| hybrid | context_precision | 0.6389 | 0.5375 | 0.7444 | 0.4449 |
| hybrid | context_recall | 0.3565 | 0.2924 | 0.4252 | 0.2695 |
| hybrid | faithfulness | 0.7836 | 0.6859 | 0.8604 | 0.3647 |
| hybrid | answer_correctness | 0.1468 | 0.1183 | 0.1782 | 0.1197 |
| hybrid_rerank | context_precision | 0.6553 | 0.5502 | 0.7619 | 0.4377 |
| hybrid_rerank | context_recall | 0.3898 | 0.3268 | 0.4588 | 0.2747 |
| hybrid_rerank | faithfulness | 0.8540 | 0.7990 | 0.9008 | 0.2128 |
| hybrid_rerank | answer_correctness | 0.1610 | 0.1341 | 0.1923 | 0.1188 |
| adaptive | context_precision | 0.6397 | 0.5366 | 0.7439 | 0.4478 |
| adaptive | context_recall | 0.3557 | 0.2887 | 0.4261 | 0.2906 |
| adaptive | faithfulness | 0.7523 | 0.6520 | 0.8376 | 0.3781 |
| adaptive | answer_correctness | 0.1556 | 0.1279 | 0.1883 | 0.1218 |

## 3. Per-Category Breakdown (Production Adaptive Pipeline)

| Category | Cases | Context Precision | Context Recall | Faithfulness | Relevancy | Correctness | Abstention | P50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exact` | 6 | 0.9722 | 0.5908 | 0.9667 | 0.5907 | 0.1505 | 100.0% | 9.3 ms |
| `conceptual` | 6 | 0.9500 | 0.6806 | 1.0000 | 0.7014 | 0.2182 | 100.0% | 7.6 ms |
| `procedural` | 6 | 0.6250 | 0.3978 | 0.9000 | 0.5226 | 0.1085 | 100.0% | 10.3 ms |
| `comparison` | 6 | 0.9535 | 0.4236 | 0.9583 | 0.7715 | 0.1364 | 100.0% | 10.6 ms |
| `summarization` | 6 | 0.7361 | 0.2568 | 0.8333 | 0.5102 | 0.1242 | 83.3% | 6.8 ms |
| `multi_part` | 6 | 0.5000 | 0.3461 | 0.7778 | 0.5075 | 0.1288 | 100.0% | 8.5 ms |
| `metadata_filtered` | 6 | 0.8188 | 0.5666 | 1.0000 | 0.5780 | 0.1282 | 100.0% | 10.5 ms |
| `multi_hop` | 6 | 0.4813 | 0.1828 | 0.8472 | 0.6541 | 0.0782 | 100.0% | 10.1 ms |
| `unanswerable` | 6 | 0.0000 | 0.0196 | 0.1667 | 0.3000 | 0.3575 | 83.3% | 8.3 ms |
| `false_premise` | 6 | 1.0000 | 0.4482 | 0.8250 | 0.5507 | 0.1380 | 100.0% | 10.0 ms |
| `adversarial` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.3130 | 0.1432 | 100.0% | 8.3 ms |