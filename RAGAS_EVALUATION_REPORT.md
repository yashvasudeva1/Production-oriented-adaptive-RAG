# Production Evaluation Report

- **Evaluation Regime:** Deterministic Offline Fallback (Heuristic Token/Set Math)
- **Evaluated At:** 2026-10-09T13:37:33.218676+00:00
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
| **DENSE** | 0.6028 | 0.3604 | 0.8227 | 0.5284 | 0.1563 | 90.9% | 830.5 ms |
| **BM25** | 0.6718 | 0.3995 | 0.9379 | 0.5484 | 0.1541 | 94.5% | 1.9 ms |
| **HYBRID** | 0.6441 | 0.3629 | 0.7727 | 0.5208 | 0.1586 | 90.0% | 5.1 ms |
| **HYBRID_RERANK** | 0.6734 | 0.4097 | 0.9803 | 0.5284 | 0.1563 | 90.9% | 131.7 ms |
| **ADAPTIVE** | 0.6368 | 0.3697 | 0.7727 | 0.5237 | 0.1617 | 90.0% | 1.9 ms |

## 2. Statistical Uncertainty (95% Bootstrap Confidence Intervals)

| Configuration | Metric | Mean | 95% CI Lower | 95% CI Upper | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| dense | context_precision | 0.6028 | 0.5193 | 0.6811 | 0.4347 |
| dense | context_recall | 0.3604 | 0.3113 | 0.4089 | 0.2581 |
| dense | faithfulness | 0.8227 | 0.7697 | 0.8742 | 0.2867 |
| dense | answer_correctness | 0.1563 | 0.1336 | 0.1783 | 0.1191 |
| bm25 | context_precision | 0.6718 | 0.5903 | 0.7415 | 0.4189 |
| bm25 | context_recall | 0.3995 | 0.3497 | 0.4443 | 0.2576 |
| bm25 | faithfulness | 0.9379 | 0.9030 | 0.9682 | 0.1732 |
| bm25 | answer_correctness | 0.1541 | 0.1316 | 0.1763 | 0.1200 |
| hybrid | context_precision | 0.6441 | 0.5563 | 0.7196 | 0.4550 |
| hybrid | context_recall | 0.3629 | 0.3063 | 0.4153 | 0.2975 |
| hybrid | faithfulness | 0.7727 | 0.6909 | 0.8545 | 0.4210 |
| hybrid | answer_correctness | 0.1586 | 0.1344 | 0.1815 | 0.1224 |
| hybrid_rerank | context_precision | 0.6734 | 0.5880 | 0.7493 | 0.4339 |
| hybrid_rerank | context_recall | 0.4097 | 0.3612 | 0.4572 | 0.2590 |
| hybrid_rerank | faithfulness | 0.9803 | 0.9652 | 0.9939 | 0.0837 |
| hybrid_rerank | answer_correctness | 0.1563 | 0.1336 | 0.1783 | 0.1191 |
| adaptive | context_precision | 0.6368 | 0.5505 | 0.7125 | 0.4504 |
| adaptive | context_recall | 0.3697 | 0.3137 | 0.4220 | 0.2974 |
| adaptive | faithfulness | 0.7727 | 0.6909 | 0.8545 | 0.4210 |
| adaptive | answer_correctness | 0.1617 | 0.1381 | 0.1852 | 0.1241 |

## 3. Per-Category Breakdown (Production Adaptive Pipeline)

| Category | Cases | Context Precision | Context Recall | Faithfulness | Relevancy | Correctness | Abstention | P50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exact` | 10 | 0.9739 | 0.6126 | 1.0000 | 0.6055 | 0.1635 | 100.0% | 2.3 ms |
| `conceptual` | 10 | 0.9963 | 0.6670 | 1.0000 | 0.7051 | 0.2566 | 100.0% | 2.0 ms |
| `procedural` | 10 | 0.5756 | 0.4052 | 0.8000 | 0.5364 | 0.0964 | 80.0% | 1.8 ms |
| `comparison` | 10 | 0.6998 | 0.4146 | 1.0000 | 0.6935 | 0.1208 | 100.0% | 2.0 ms |
| `summarization` | 10 | 0.6900 | 0.3245 | 0.9000 | 0.5380 | 0.1120 | 90.0% | 1.8 ms |
| `multi_part` | 10 | 0.6833 | 0.4055 | 0.9000 | 0.5071 | 0.1464 | 90.0% | 1.9 ms |
| `metadata_filtered` | 10 | 0.7960 | 0.4652 | 0.9000 | 0.5403 | 0.1293 | 90.0% | 1.7 ms |
| `multi_hop` | 10 | 0.6465 | 0.2947 | 0.9000 | 0.5043 | 0.0955 | 90.0% | 1.7 ms |
| `unanswerable` | 10 | 0.0000 | 0.0000 | 0.1000 | 0.3272 | 0.3900 | 90.0% | 1.8 ms |
| `false_premise` | 10 | 0.9432 | 0.4771 | 1.0000 | 0.4956 | 0.1275 | 100.0% | 1.7 ms |
| `adversarial` | 10 | 0.0000 | 0.0000 | 0.0000 | 0.3078 | 0.1404 | 60.0% | 1.9 ms |