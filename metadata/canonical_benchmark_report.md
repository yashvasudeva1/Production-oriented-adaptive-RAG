# Nexus Adaptive RAG: Canonical Engineering Benchmark Report

- **Evaluation Run ID:** `eval_run_1794936962`
- **Timestamp:** `2026-11-17T17:36:02Z`
- **Git Commit SHA:** `c7ef06329d289f28249f8de8dd7314abf06cf91d`
- **Dataset Version:** 110 held-out cases (90 answerable, 20 unanswerable/adversarial)
- **Dataset Hash (SHA-256):** `ffe5e543d85301e83b59efe89f299b5a054a988d9b8bece0e8a124dc3505fa05`
- **Relevance Reference Hash (SHA-256):** `b007c78e86f36ce77a50e80d606b857262bef7c00a4f74e8b92b414c344e6889`
- **Corpus Hash (130 Chunks, SHA-256):** `13d4da4eddf022fdb6ad31b357fb0ac4499d44e88dca71e7f45426495ec5980e`
- **Evaluator Implementation:** `DeterministicEvaluatorEngine v2.1` (`src.evaluation.ragas_metrics`)
- **Evaluation Mode:** Deterministic Offline Grounded (Local CPU)
- **Cache State:** Cold cache per query (`CacheManager.clear()`)
- **Latency Scope:** Full end-to-end pipeline latency per query in milliseconds (retrieval + fusion + rerank + extractive synthesis + validation)

---

## 1. Metric Definitions & Audit Specifications

1. **Recall@K (Answerable Subset, N=90):**
   Proportion of answerable queries where at least one ground-truth chunk or keyword-matching evidence passage appears in the top-K retrieved candidates.
   $$\text{Recall@K} = \frac{1}{|Q_{\text{ans}}|} \sum_{q \in Q_{\text{ans}}} \mathbb{I}(\text{hits} \cap \text{topK} \ne \emptyset)$$
2. **Recall@K (Blended Suite, N=110):**
   Evaluation across all 110 suite cases. For unanswerable/adversarial queries (where no relevant document exists in the corpus), a correct abstention is credited as 1.0 (and 0.0 on failed abstention).
3. **MRR (Mean Reciprocal Rank):**
   Reciprocal rank of the first relevant retrieved chunk ($1/\text{rank}_1$). For unanswerable queries in the blended suite, credited as 1.0 on correct abstention.
4. **nDCG@K (Normalized Discounted Cumulative Gain):**
   Measures ranking quality with logarithmic position discounting against ideal DCG.
5. **Answer Correctness (Deterministic Token-Set F1 Proxy):**
   Computes harmonic mean of precision and recall over informative non-stopword tokens between generated text and canonical reference answer:
   $$\text{F1} = \frac{2 \cdot P \cdot R}{P + R}, \quad P = \frac{|T_{\text{pred}} \cap T_{\text{ref}}|}{|T_{\text{pred}}|}, \quad R = \frac{|T_{\text{pred}} \cap T_{\text{ref}}|}{|T_{\text{ref}}|}$$
   *Audit Distinction:* This is a deterministic offline lexical proxy and is NOT identical to LLM-judged RAGAS statement-level factual correctness.
6. **Cross-Encoder Invocation & Bypass Rate:**
   $$\text{Invocation Rate} = \frac{\text{queries invoking cross-encoder}}{\text{total queries}}, \quad \text{Bypass Rate} = 1.0 - \text{Invocation Rate}$$
   Reported as N/A for architectures without a cross-encoder stage.
7. **Latency Distributions:**
   Reported as median (P50), P95, and P99 tail percentiles to measure tail behavior.

---

## 2. Standard IR Benchmark: Answerable Queries Only (N=90)

The standard Information Retrieval (IR) evaluation strictly isolates the 90 queries with positive ground-truth relevance labels:

| Configuration | Recall@5 | Prec@5 | MRR | nDCG@5 | Answer Correctness (F1) | Faithfulness | Citation Acc | P50 Latency | P95 Latency | Cross-Encoder Bypass Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | 1.0000 | 0.7200 | 0.9704 | 0.9246 | 0.4086 | 0.9327 | 1.0000 | 5.4 ms | 7.5 ms | N/A |
| **DENSE** | 0.9000 | 0.4422 | 0.8537 | 0.8374 | 0.3453 | 0.9232 | 1.0000 | 4.2 ms | 7.5 ms | N/A |
| **HYBRID** | 0.9222 | 0.4911 | 0.7313 | 0.6600 | 0.2930 | 0.9364 | 1.0000 | 16.7 ms | 29.1 ms | N/A |
| **HYBRID_RERANK** | 0.9667 | 0.6300 | 0.9250 | 0.8805 | 0.3447 | 0.9327 | 1.0000 | 16.5 ms | 27.0 ms | 0.0% |
| **SIMPLIFIED_ADAPTIVE_NO_GUARD** | 0.9667 | 0.6389 | 0.9361 | 0.9061 | 0.3450 | 0.9327 | 1.0000 | 8.0 ms | 20.9 ms | 6.4% |
| **SIMPLIFIED_ADAPTIVE_GUARD** | **0.9667** | **0.6256** | **0.9287** | **0.8885** | **0.3396** | **0.9327** | **1.0000** | **7.7 ms** | **11.5 ms** | **12.7%** |

---

## 3. Controlled 6-Way Ablation Table: Blended Suite (N=110)

Evaluates end-to-end behavior across all 110 suite cases (90 answerable + 20 unanswerable/adversarial credited on abstention):

| Configuration | Recall@5 (95% CI) | Prec@5 | MRR | nDCG@5 | Answer Correctness | Faithfulness | Citation Acc | Abstention Qual | Bypass Rate | P50 / P95 / P99 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | 0.991 [0.964, 1.000] | 0.771 | 0.967 | 0.929 | 0.409 | 0.933 | 1.000 | 0.991 | N/A | 5.4ms / 7.5ms / 8.2ms |
| **DENSE** | 0.909 [0.855, 0.955] | 0.534 | 0.871 | 0.858 | 0.345 | 0.923 | 1.000 | 0.918 | N/A | 4.2ms / 7.5ms / 9.2ms |
| **HYBRID** | 0.918 [0.864, 0.973] | 0.575 | 0.762 | 0.704 | 0.293 | 0.936 | 1.000 | 0.973 | N/A | 16.7ms / 29.1ms / 31.0ms |
| **HYBRID_RERANK** | 0.964 [0.927, 0.991] | 0.689 | 0.929 | 0.893 | 0.345 | 0.933 | 1.000 | 0.982 | 0.0% | 16.5ms / 27.0ms / 28.5ms |
| **SIMPLIFIED_ADAPTIVE_NO_GUARD** | 0.964 [0.927, 0.991] | 0.696 | 0.939 | 0.914 | 0.345 | 0.933 | 1.000 | 0.982 | 6.4% | 8.0ms / 20.9ms / 23.9ms |
| **SIMPLIFIED_ADAPTIVE_GUARD** | 0.964 [0.927, 0.991] | 0.685 | 0.933 | 0.900 | 0.340 | 0.933 | 1.000 | 0.982 | 12.7% | **7.7ms / 11.5ms / 21.6ms** |

---

## 4. Statistical Equivalence & Significance Testing

### A. Simplified Adaptive (With Guard) vs. Unconditional Hybrid Rerank

Formal Two One-Sided Tests (TOST) within a predefined margin of $\delta = \pm 0.03$:

| Metric | Adaptive Guard | Hybrid Rerank | Difference | Paired t-test ($p$) | Wilcoxon ($p$) | TOST Equivalence ($p_{\text{TOST}}$) | Statistical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Recall@5** | 0.9636 | 0.9636 | 0.0000 | 1.000 | 1.000 | 0.000 | **Formally Equivalent** ($p < 0.05$) |
| **Precision@5** | 0.6855 | 0.6891 | -0.0036 | 0.595 | 0.776 | 9.54e-05 | **Formally Equivalent** ($p < 0.05$) |
| **MRR** | 0.9326 | 0.9295 | +0.0031 | 0.733 | 1.000 | 1.48e-03 | **Formally Equivalent** ($p < 0.05$) |
| **nDCG@5** | 0.8997 | 0.8931 | +0.0066 | 0.518 | 0.322 | 1.08e-02 | **Formally Equivalent** ($p < 0.05$) |
| **Answer Correctness** | 0.3396 | 0.3447 | -0.0051 | 0.083 | 0.079 | 2.96e-14 | **Formally Equivalent** ($p < 0.05$) |
| **Faithfulness** | 0.9327 | 0.9327 | 0.0000 | 1.000 | 1.000 | 4.41e-21 | **Formally Equivalent** ($p < 0.05$) |
| **End-to-End Latency** | 7.79 ms | 14.86 ms | **-7.07 ms** | **1.10e-17** | **1.24e-13** | N/A ($p > 0.05$) | **Statistically Faster** ($p < 10^{-16}$) |

### B. Top-1 Consensus Guard Ablation (Adaptive Guard vs. Adaptive No-Guard)

| Metric | With Guard | Without Guard | Absolute Difference | TOST Status (±0.03 Margin) | Engineering Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Recall@5** | 0.9636 | 0.9636 | 0.0000 | Strictly Equivalent ($p = 0.000$) | Zero recall loss |
| **MRR** | 0.9326 | 0.9386 | -0.0060 | Strictly Equivalent ($p = 6.95\text{e-}05$) | Ranking preserved |
| **nDCG@5** | 0.8997 | 0.9141 | -0.0144 | Strictly Equivalent ($p = 1.52\text{e-}02$) | Cumulative gain preserved |
| **Answer Correctness** | 0.3396 | 0.3450 | -0.0054 | Strictly Equivalent ($p = 3.33\text{e-}16$) | Answer accuracy preserved |
| **P95 Latency** | **11.5 ms** | 20.9 ms | **-9.4 ms** | Statistically Faster | **1.82× faster tail latency** |

---

## 5. Audit of Previous Benchmark Discrepancies

1. **Root Cause of BM25 & Simplified Adaptive Metric Shifts:**
   - In the earlier November 17 17:02 evaluation, `CitationValidator._check_polarity_inversion` checked antonyms across all concatenated context chunks (`all_ctx_text`), causing 11 answerable queries to fail validation and return `sources=[]`.
   - Because `sources` was empty, `rel_scores` was empty, driving `recall_at_k` and `mrr` to 0.0 on those queries.
   - Once scoped strictly to the cited `evidence_span`, those queries returned their retrieved sources, lifting answerable Recall@5 from 73/90 (0.8111) to 90/90 (1.0000) for BM25, and from 68/90 (0.7556) to 87/90 (0.9667) for Simplified Adaptive.
2. **Blended Metric Reporting:**
   - When blended across all 110 suite queries (crediting 19 successful abstentions as 1.0), BM25 rose from $92/110 = 0.836$ to $109/110 = 0.991$, and Simplified Adaptive rose from $87/110 = 0.791$ to $106/110 = 0.964$.
   - Reporting both the pure 90-case answerable IR metrics and the 110-case blended metrics guarantees complete transparency.
