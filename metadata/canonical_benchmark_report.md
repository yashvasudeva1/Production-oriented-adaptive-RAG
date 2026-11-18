# Nexus Adaptive RAG: Canonical Engineering Benchmark Report

- **Evaluation Run ID:** `eval_run_1795012088`
- **Timestamp:** `2026-11-18T14:28:08Z`
- **Git Commit SHA:** `18ade9500f9f44157c39482dbd5d995b515155dc`
- **Dataset:** 110 held-out cases (90 answerable, 20 unanswerable/adversarial)
- **Dataset Hash (SHA-256):** `a9f91d3e83dd8bf1e724ad6744eec01215e6f4ab4ad04039f1370b8ef72d37fc`
- **References Hash (SHA-256):** `b007c78e86f36ce77a50e80d606b857262bef7c00a4f74e8b92b414c344e6889`
- **Corpus Hash (130 Chunks, SHA-256):** `52f653d9dc05265ed276f968bba11dd519d55f483ed76475b38ce7635c3905de`
- **Evaluator:** `DeterministicEvaluatorEngine v2.1 (src.evaluation.ragas_metrics)`
- **Embedder Backend:** `MockEmbedder(dim=128)`
- **Reranker Backend:** `lexical_fallback`
- **RRF Configuration:** `k=20`, default `0.50 BM25 / 0.50 Dense`, identifier boost `0.55 BM25 / 0.45 Dense`
- **CI Methodology:** Wilson score 95% CI for binary proportions (Recall@K, Abstention); 10,000-resample percentile bootstrap 95% CI (seed=42) for continuous metrics & paired differences
- **Cache State:** Cold cache per query (CacheManager.clear() + generator._gen_cache.clear())
- **Latency Scope:** Full end-to-end pipeline latency per query in milliseconds (retrieval + fusion + rerank + synthesis + validation)

> [!WARNING]
> **Explicit Offline & Synthetic Benchmark Limitations:**
> 1. **130-Chunk Synthetic Corpus:** The corpus consists of 90 reference-derived target chunks plus 40 synthetic distractors. High Recall@5 (~0.93–1.00) reflects this small controlled corpus and **does not** predict recall on a production 100k–10M chunk corpus.
> 2. **MockEmbedder (`dim=128`):** Dense retrieval runs on deterministic MD5 token-hash projections, not a trained neural transformer encoder. Semantic synonymy and paraphrase retrieval are **not** tested in offline mode.
> 3. **Lexical Reranker Fallback:** Because `cross-encoder/ms-marco-MiniLM-L-6-v2` weights are not cached locally in this offline environment, `CrossEncoderReranker` runs a ~0.2ms lexical Jaccard fallback instead of a ~150–200ms neural cross-encoder. Therefore, **offline latency measurements cannot validate real neural cross-encoder bypass savings**.
> 4. **Semantic Evaluation Status:** Neural Embedding Cosine = `NOT_EXECUTED` (Local neural weights unavailable (OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files. Check your internet connectio)); LLM-as-a-Judge RAGAS = `NOT_EXECUTED` (OFFLINE_EVAL=1 active). Unexecuted semantic metrics are **never** fabricated or reported as completed.

---

## 0. Evaluation Execution Classification

| Evaluation Track | Execution Status | Backend / Corpus | Interpretability Scope |
| :--- | :---: | :--- | :--- |
| **Canonical 110-Query Retrieval (BM25)** | **EXECUTED (Real Lexical Model, Synthetic Corpus)** | `BM25Okapi` on 130-chunk synthetic corpus | Real lexical retriever; synthetic small corpus |
| **Canonical 110-Query Retrieval (Dense, Hybrid, Rerank, Adaptive)** | **EXECUTED (Offline Synthetic)** | `MockEmbedder(dim=128)` + `lexical_fallback` (~0.2ms) + `OfflineExtractiveProvider` | Pipeline plumbing, fusion math, gating, and abstention verification only; **not** neural quality or real CE latency |
| **Real-Model Neural Suite (`--mode neural`)** | **NOT_EXECUTED** | Target: `BAAI/bge-small-en-v1.5` + `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local HuggingFace weights unavailable offline; zero mock substitution |
| **Semantic Answer Eval (Embedding Cosine & LLM-as-a-Judge RAGAS)** | **NOT_EXECUTED** | Target: Neural SentenceTransformer + Online LLM Judge | Offline mode (`OFFLINE_EVAL=1`) & no cached weights |
| **Google FRAMES Multi-Hop (`FRAMES_EVALUATION_REPORT.md`)** | **HISTORICAL / SIMULATED** | Historical live runs (550-chunk Wikipedia corpus) + simulated `SIMPLIFIED_ADAPTIVE` replay | Raw 550-chunk corpus not packaged; `SIMPLIFIED_ADAPTIVE` row is simulation replay only |

---

## 1. Metric Definitions & Evaluation Protocol

1. **Answerable-Only Retrieval Metrics (Primary, N=90):** Recall@5, Precision@5, MRR, and nDCG@5 are computed strictly on the 90 answerable queries using **pre-generation retrieved candidates** (`resp.retrieval.retrieved_candidates`), decoupled from citation validation and abstention. Unanswerable queries have no positive relevance labels and are **never** assigned Recall@K=1 or MRR=1.
2. **Beyond-Unigram Answer Evaluation (Executed Offline, Answerable N=90):**
   - **Token-Set F1:** Unweighted non-stopword unigram F1 proxy.
   - **Claim Factual Alignment F1:** Atomic clause decomposition requiring >=50% predicate coverage AND exact numeric/entity consistency per matched claim.
   - **Entity/Numeric F1:** Exact fidelity across numbers, percentages, formulas, acronyms, and named entities.
   - **Phrase Bigram F1:** Ordered non-stopword bigram F1 (ROUGE-2 style).
3. **Unanswerable & Adversarial Abstention Metrics (Separate, N=20 & N=110):** True Abstention Rate (N=20), False Abstention Rate on Answerable (N=90), and Overall Abstention Accuracy (N=110).
4. **Confidence Intervals & Zero-Variance Discipline:** Binary proportions (`Recall@5`, `Unanswerable Abstention`) use **Wilson score 95% CIs** (non-degenerate at 0% and 100%). Continuous metrics (`MRR`, `nDCG@5`, `F1`, `Latency`) and paired per-query differences use **10,000 deterministic bootstrap resamples** (`seed=42`, 2.5th–97.5th percentiles). Zero-variance paired comparisons (`0` discordant pairs) report `N/A (0 var)` for parametric tests and `Observed Equal (0 var; TOST N/A)` rather than claiming statistically established population equivalence.
5. **Multiple-Testing Control (Holm-Bonferroni):** Exploratory pairwise hypothesis tests in Section 5 are grouped into three explicitly defined comparison families (`retrieval_quality`, `generation_quality`, `latency`). Both unadjusted $p$-values and Holm-Bonferroni adjusted $p$-values (controlling FWER at $\alpha=0.05$ within each family) are reported side by side.

---

## 2. Primary Retrieval Quality (Answerable Queries Only, N=90, Pre-Generation)

| Configuration | Recall@5 (Wilson 95% CI) | Prec@5 | MRR (Boot 95% CI) | nDCG@5 (Boot 95% CI) | CE Inv / Byp | Mean CE Pairs | P50 / P95 / P99 Latency (All 110) |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **BM25** | 1.0000 [0.9591, 1.0000] | 0.7311 | 0.9704 [0.9426, 0.9944] | 0.9246 [0.8989, 0.9478] | N/A | N/A | 5.3ms / 7.4ms / 9.2ms |
| **DENSE** | 0.9333 [0.8621, 0.9691] | 0.4711 | 0.8870 [0.8259, 0.9389] | 0.8678 [0.8105, 0.9169] | N/A | N/A | 3.8ms / 6.2ms / 6.7ms |
| **HYBRID** | 1.0000 [0.9591, 1.0000] | 0.6600 | 0.9704 [0.9426, 0.9944] | 0.9191 [0.8904, 0.9444] | N/A | N/A | 5.2ms / 10.8ms / 11.8ms |
| **HYBRID_RERANK** | 1.0000 [0.9591, 1.0000] | 0.6711 | 0.9648 [0.9333, 0.9889] | 0.9254 [0.9004, 0.9481] | 100.0% / 0.0% | 27.13 | 5.5ms / 11.2ms / 12.3ms |
| **SIMPLIFIED_ADAPTIVE_NO_GUARD** | 1.0000 [0.9591, 1.0000] | 0.6622 | 0.9704 [0.9426, 0.9944] | 0.9418 [0.9194, 0.9615] | 96.4% / 3.6% | 7.56 | 5.2ms / 7.5ms / 8.6ms |
| **SIMPLIFIED_ADAPTIVE_GUARD** | 1.0000 [0.9591, 1.0000] | 0.6622 | 0.9704 [0.9426, 0.9944] | 0.9418 [0.9194, 0.9617] | 91.8% / 8.2% | 7.29 | 5.3ms / 7.2ms / 7.7ms |
| **SIMPLIFIED_ADAPTIVE_INTENT_BUDGET** | 1.0000 [0.9591, 1.0000] | 0.6644 | 0.9704 [0.9426, 0.9944] | 0.9376 [0.9139, 0.9589] | 91.8% / 8.2% | 8.49 | 5.3ms / 8.2ms / 8.6ms |

---

## 3. Generation, Claim-Level Alignment & Grounding Evaluation (Answerable N=90)

| Configuration | Token F1 | Claim Factual F1 | Entity/Numeric F1 | Bigram F1 | Faithfulness | Citation Acc | Semantic Cosine / LLM Judge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | 0.7410 | 0.6640 | 0.7095 | 0.6812 | 0.9963 | 1.0000 | NOT_EXECUTED (offline) |
| **DENSE** | 0.6443 | 0.5666 | 0.6211 | 0.5813 | 0.9963 | 1.0000 | NOT_EXECUTED (offline) |
| **HYBRID** | 0.7284 | 0.6492 | 0.7009 | 0.6695 | 0.9963 | 1.0000 | NOT_EXECUTED (offline) |
| **HYBRID_RERANK** | 0.7500 | 0.6803 | 0.7154 | 0.6982 | 0.9963 | 1.0000 | NOT_EXECUTED (offline) |
| **SIMPLIFIED_ADAPTIVE_NO_GUARD** | 0.7262 | 0.6492 | 0.6987 | 0.6689 | 0.9963 | 1.0000 | NOT_EXECUTED (offline) |
| **SIMPLIFIED_ADAPTIVE_GUARD** | 0.7262 | 0.6492 | 0.6987 | 0.6689 | 0.9963 | 1.0000 | NOT_EXECUTED (offline) |
| **SIMPLIFIED_ADAPTIVE_INTENT_BUDGET** | 0.7262 | 0.6492 | 0.6987 | 0.6689 | 0.9963 | 1.0000 | NOT_EXECUTED (offline) |

---

## 4. Abstention Quality (Reported Separately from Retrieval Metrics)

| Configuration | Unanswerable True Abstention (N=20, Wilson 95% CI) | Answerable False Abstention (N=90) | Overall Abstention Accuracy (N=110) |
| :--- | :--- | :--- | :--- |
| **BM25** | 1.0000 [0.8389, 1.0000] | 0.0000 | 1.0000 |
| **DENSE** | 1.0000 [0.8389, 1.0000] | 0.0667 | 0.9455 |
| **HYBRID** | 1.0000 [0.8389, 1.0000] | 0.0000 | 1.0000 |
| **HYBRID_RERANK** | 1.0000 [0.8389, 1.0000] | 0.0000 | 1.0000 |
| **SIMPLIFIED_ADAPTIVE_NO_GUARD** | 1.0000 [0.8389, 1.0000] | 0.0000 | 1.0000 |
| **SIMPLIFIED_ADAPTIVE_GUARD** | 1.0000 [0.8389, 1.0000] | 0.0000 | 1.0000 |
| **SIMPLIFIED_ADAPTIVE_INTENT_BUDGET** | 1.0000 [0.8389, 1.0000] | 0.0000 | 1.0000 |

---

## 5. Paired Per-Query Statistical Comparisons (Adaptive Guard Baseline vs. All Configurations)

- **Explicit Delta Convention (All Section 5 Tables):** $\Delta = \text{SIMPLIFIED\_ADAPTIVE\_GUARD} - \text{Comparator}$. Positive values mean `SIMPLIFIED_ADAPTIVE_GUARD` is higher than the comparator; negative values mean `SIMPLIFIED_ADAPTIVE_GUARD` is lower (for latency, negative means faster).
- **Uncertainty & Equivalence:** 95% paired bootstrap CIs (`B=10,000`, `seed=42`), paired $t$-tests, Wilcoxon signed-rank tests, and TOST equivalence ($\delta = \pm 0.03$).
- **Observed Equality vs. Statistical Equivalence:** When all paired observations are identical (`0` discordant pairs, zero variance), parametric tests cannot run: tables report `N/A (0 var)` and `Observed Equal (0 var; TOST N/A)`. Only non-zero-variance comparisons with $p_{\text{TOST}} < 0.05$ are labeled `Stat. Equiv`.
- **Holm-Bonferroni FWER Adjustment:** Both unadjusted $p$ and Holm-adjusted $p_{\text{Holm}}$ are shown (`p / p_Holm`), adjusted separately within each documented comparison family below (zero-variance slots excluded from active test count $m$).

### Documented Holm-Bonferroni Comparison Families (Across All 6 Comparators)

| Comparison Family | Metrics Included | Total Slots | Active Tests $m$ (t / W) | Zero-Var Excluded (t / W) | Sig. at $\alpha=0.05$ Unadj (t / W) | Sig. at $\alpha=0.05$ Holm (t / W) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **retrieval_quality** | `recall_at_k`, `mrr`, `ndcg_at_k` | 18 | 10 / 10 | 8 / 8 | 5 / 5 | 3 / 2 |
| **generation_quality** | `answer_correctness`, `claim_factual_f1`, `entity_numeric_f1`, `faithfulness` | 24 | 11 / 11 | 13 / 13 | 3 / 3 | 2 / 3 |
| **latency** | `latency_ms` | 6 | 6 / 6 | 0 / 0 | 3 / 3 | 3 / 3 |

### SIMPLIFIED_ADAPTIVE_GUARD vs. BM25

| Metric | Family | N | Guard Mean | Comp Mean | Paired Diff (Guard - Comp, 95% Boot CI, B=10k) | Paired t (p / Holm) | Wilcoxon (p / Holm) | TOST (+/-0.03) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **recall_at_k** | `retrieval_quality` | 90 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **mrr** | `retrieval_quality` | 90 | 0.9704 | 0.9704 | +0.0000 [-0.0222, +0.0222] | 1.000e+00 / 1.000e+00 | 1.000e+00 / 1.000e+00 | Stat. Equiv (p=4.33e-03) |
| **ndcg_at_k** | `retrieval_quality` | 90 | 0.9418 | 0.9246 | +0.0172 [-0.0050, +0.0407] | 1.496e-01 / 7.481e-01 | 1.912e-01 / 9.558e-01 | Not Equiv (p=1.40e-01) |
| **answer_correctness** | `generation_quality` | 90 | 0.7262 | 0.7410 | -0.0148 [-0.0555, +0.0236] | 4.640e-01 / 1.000e+00 | 4.631e-01 / 1.000e+00 | Not Equiv (p=2.26e-01) |
| **claim_factual_f1** | `generation_quality` | 90 | 0.6492 | 0.6640 | -0.0148 [-0.0711, +0.0400] | 6.007e-01 / 1.000e+00 | 6.875e-01 / 1.000e+00 | Not Equiv (p=2.96e-01) |
| **entity_numeric_f1** | `generation_quality` | 90 | 0.6987 | 0.7095 | -0.0108 [-0.0546, +0.0307] | 6.213e-01 / 1.000e+00 | 5.781e-01 / 1.000e+00 | Not Equiv (p=1.89e-01) |
| **faithfulness** | `generation_quality` | 90 | 0.9963 | 0.9963 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **latency_ms** | `latency` | 110 | 5.2852 | 5.1971 | +0.0882 [-0.2073, +0.3785] | 5.599e-01 / 1.000e+00 | 2.406e-01 / 6.559e-01 | Not Equiv (p=6.50e-01) |

### SIMPLIFIED_ADAPTIVE_GUARD vs. DENSE

| Metric | Family | N | Guard Mean | Comp Mean | Paired Diff (Guard - Comp, 95% Boot CI, B=10k) | Paired t (p / Holm) | Wilcoxon (p / Holm) | TOST (+/-0.03) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **recall_at_k** | `retrieval_quality` | 90 | 1.0000 | 0.9333 | +0.0667 [+0.0222, +0.1222] | 1.347e-02 / 8.083e-02 | 3.125e-02 / 1.875e-01 | Not Equiv (p=9.16e-01) |
| **mrr** | `retrieval_quality` | 90 | 0.9704 | 0.8870 | +0.0833 [+0.0333, +0.1389] | 2.260e-03 / 2.260e-02 | 2.441e-03 / 2.441e-02 | Not Equiv (p=9.76e-01) |
| **ndcg_at_k** | `retrieval_quality` | 90 | 0.9418 | 0.8678 | +0.0740 [+0.0299, +0.1263] | 3.625e-03 / 3.262e-02 | 9.689e-03 / 7.751e-02 | Not Equiv (p=9.61e-01) |
| **answer_correctness** | `generation_quality` | 90 | 0.7262 | 0.6443 | +0.0819 [+0.0358, +0.1349] | 1.737e-03 / 1.910e-02 | 1.402e-03 / 1.542e-02 | Not Equiv (p=9.78e-01) |
| **claim_factual_f1** | `generation_quality` | 90 | 0.6492 | 0.5666 | +0.0826 [+0.0274, +0.1452] | 6.475e-03 / 5.828e-02 | 4.395e-03 / 3.955e-02 | Not Equiv (p=9.60e-01) |
| **entity_numeric_f1** | `generation_quality` | 90 | 0.6987 | 0.6211 | +0.0777 [+0.0318, +0.1327] | 3.379e-03 / 3.379e-02 | 3.309e-03 / 3.309e-02 | Not Equiv (p=9.66e-01) |
| **faithfulness** | `generation_quality` | 90 | 0.9963 | 0.9963 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **latency_ms** | `latency` | 110 | 5.2852 | 3.7169 | +1.5684 [+1.3929, +1.7462] | 2.178e-33 / 1.307e-32 | 3.417e-19 / 2.050e-18 | Not Equiv (p=1.00e+00) |

### SIMPLIFIED_ADAPTIVE_GUARD vs. HYBRID

| Metric | Family | N | Guard Mean | Comp Mean | Paired Diff (Guard - Comp, 95% Boot CI, B=10k) | Paired t (p / Holm) | Wilcoxon (p / Holm) | TOST (+/-0.03) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **recall_at_k** | `retrieval_quality` | 90 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **mrr** | `retrieval_quality` | 90 | 0.9704 | 0.9704 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **ndcg_at_k** | `retrieval_quality` | 90 | 0.9418 | 0.9191 | +0.0227 [+0.0088, +0.0393] | 4.605e-03 / 3.684e-02 | 5.311e-03 / 4.780e-02 | Not Equiv (p=1.77e-01) |
| **answer_correctness** | `generation_quality` | 90 | 0.7262 | 0.7284 | -0.0022 [-0.0066, +0.0000] | 3.200e-01 / 1.000e+00 | 1.000e+00 / 1.000e+00 | Stat. Equiv (p=7.49e-26) |
| **claim_factual_f1** | `generation_quality` | 90 | 0.6492 | 0.6492 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **entity_numeric_f1** | `generation_quality` | 90 | 0.6987 | 0.7009 | -0.0022 [-0.0067, +0.0000] | 3.200e-01 / 1.000e+00 | 1.000e+00 / 1.000e+00 | Stat. Equiv (p=1.84e-25) |
| **faithfulness** | `generation_quality` | 90 | 0.9963 | 0.9963 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **latency_ms** | `latency` | 110 | 5.2852 | 6.2936 | -1.0084 [-1.3479, -0.6905] | 2.341e-08 / 9.363e-08 | 1.868e-06 / 7.472e-06 | Not Equiv (p=1.00e+00) |

### SIMPLIFIED_ADAPTIVE_GUARD vs. HYBRID_RERANK

| Metric | Family | N | Guard Mean | Comp Mean | Paired Diff (Guard - Comp, 95% Boot CI, B=10k) | Paired t (p / Holm) | Wilcoxon (p / Holm) | TOST (+/-0.03) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **recall_at_k** | `retrieval_quality` | 90 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **mrr** | `retrieval_quality` | 90 | 0.9704 | 0.9648 | +0.0056 [+0.0000, +0.0167] | 3.200e-01 / 1.000e+00 | 1.000e+00 / 1.000e+00 | Stat. Equiv (p=1.50e-05) |
| **ndcg_at_k** | `retrieval_quality` | 90 | 0.9418 | 0.9254 | +0.0164 [+0.0048, +0.0294] | 9.919e-03 / 6.943e-02 | 1.149e-02 / 8.043e-02 | Stat. Equiv (p=1.54e-02) |
| **answer_correctness** | `generation_quality` | 90 | 0.7262 | 0.7500 | -0.0238 [-0.0547, +0.0000] | 8.789e-02 / 6.788e-01 | 2.500e-01 / 1.000e+00 | Not Equiv (p=3.26e-01) |
| **claim_factual_f1** | `generation_quality` | 90 | 0.6492 | 0.6803 | -0.0311 [-0.0711, +0.0000] | 8.484e-02 / 6.788e-01 | 2.500e-01 / 1.000e+00 | Not Equiv (p=5.25e-01) |
| **entity_numeric_f1** | `generation_quality` | 90 | 0.6987 | 0.7154 | -0.0167 [-0.0444, +0.0000] | 1.812e-01 / 1.000e+00 | 5.000e-01 / 1.000e+00 | Not Equiv (p=1.42e-01) |
| **faithfulness** | `generation_quality` | 90 | 0.9963 | 0.9963 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **latency_ms** | `latency` | 110 | 5.2852 | 6.4123 | -1.1271 [-1.4845, -0.7900] | 5.002e-09 / 2.501e-08 | 1.720e-08 / 8.600e-08 | Not Equiv (p=1.00e+00) |

### SIMPLIFIED_ADAPTIVE_GUARD vs. SIMPLIFIED_ADAPTIVE_NO_GUARD

| Metric | Family | N | Guard Mean | Comp Mean | Paired Diff (Guard - Comp, 95% Boot CI, B=10k) | Paired t (p / Holm) | Wilcoxon (p / Holm) | TOST (+/-0.03) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **recall_at_k** | `retrieval_quality` | 90 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **mrr** | `retrieval_quality` | 90 | 0.9704 | 0.9704 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **ndcg_at_k** | `retrieval_quality` | 90 | 0.9418 | 0.9418 | +0.0000 [-0.0007, +0.0008] | 9.255e-01 / 1.000e+00 | 1.000e+00 / 1.000e+00 | Stat. Equiv (p=1.08e-87) |
| **answer_correctness** | `generation_quality` | 90 | 0.7262 | 0.7262 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **claim_factual_f1** | `generation_quality` | 90 | 0.6492 | 0.6492 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **entity_numeric_f1** | `generation_quality` | 90 | 0.6987 | 0.6987 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **faithfulness** | `generation_quality` | 90 | 0.9963 | 0.9963 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **latency_ms** | `latency` | 110 | 5.2852 | 5.2591 | +0.0261 [-0.1009, +0.1495] | 6.858e-01 / 1.000e+00 | 2.186e-01 / 6.559e-01 | Not Equiv (p=4.76e-01) |

### SIMPLIFIED_ADAPTIVE_GUARD vs. SIMPLIFIED_ADAPTIVE_INTENT_BUDGET

| Metric | Family | N | Guard Mean | Comp Mean | Paired Diff (Guard - Comp, 95% Boot CI, B=10k) | Paired t (p / Holm) | Wilcoxon (p / Holm) | TOST (+/-0.03) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **recall_at_k** | `retrieval_quality` | 90 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **mrr** | `retrieval_quality` | 90 | 0.9704 | 0.9704 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **ndcg_at_k** | `retrieval_quality` | 90 | 0.9418 | 0.9376 | +0.0042 [-0.0032, +0.0130] | 3.138e-01 / 1.000e+00 | 4.375e-01 / 1.000e+00 | Stat. Equiv (p=8.83e-09) |
| **answer_correctness** | `generation_quality` | 90 | 0.7262 | 0.7262 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **claim_factual_f1** | `generation_quality` | 90 | 0.6492 | 0.6492 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **entity_numeric_f1** | `generation_quality` | 90 | 0.6987 | 0.6987 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **faithfulness** | `generation_quality` | 90 | 0.9963 | 0.9963 | +0.0000 [+0.0000, +0.0000] | N/A (0 var) | N/A (0 var) | Observed Equal (0 var; TOST N/A) |
| **latency_ms** | `latency` | 110 | 5.2852 | 5.4247 | -0.1395 [-0.2957, +0.0065] | 7.163e-02 / 2.149e-01 | 5.499e-01 / 6.559e-01 | Not Equiv (p=9.22e-01) |

---

## 6. Adaptive Reranking Experiment: Intent-Aware Candidate Budget (`[:6]` vs `[:10]`)

- **Explicit Delta Convention (Section 6):** $\Delta = \text{Experimental (SIMPLIFIED\_ADAPTIVE\_INTENT\_BUDGET)} - \text{Baseline (SIMPLIFIED\_ADAPTIVE\_GUARD)}$. Negative quality deltas indicate the experimental budget regressed relative to baseline (opposite sign of Section 5's `Guard - Comp` convention).
- **Baseline (`SIMPLIFIED_ADAPTIVE_GUARD`):** `802` total CE candidate pairs scored across 110 queries (`7.29` pairs/query).
- **Experimental (`SIMPLIFIED_ADAPTIVE_INTENT_BUDGET`):** `934` total CE candidate pairs scored across 110 queries (`8.49` pairs/query).
- **Additional Reranker Inference Cost (Exp - Base):** `+132` pairs (`+16.5%` cross-encoder pair evaluations).
- **Aggregate nDCG@5 Delta (Exp - Base, N=90):** `-0.0042` (95% Boot CI `[-0.0130, +0.0032]`; equivalently `Guard - IntentBudget = +0.0042 [-0.0032, +0.0130]` in Section 5).
- **Answerable Queries with Quality Changes:** `6` / 90 (`0` changes in Recall@5, MRR, Token F1, or Claim F1).

| Query ID | Category | Planned Intent | Pool (Base -> Exp) | MRR (Base -> Exp) | nDCG@5 (Base -> Exp) | nDCG@5 Δ (Exp - Base) | Token F1 (Base -> Exp) | Claim F1 (Base -> Exp) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `concept_10` | conceptual | detailed | 6 -> 10 | 1.00 -> 1.00 | 0.9675 -> 1.0000 | +0.0325 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 |
| `comp_06` | comparison | comparison | 6 -> 10 | 1.00 -> 1.00 | 0.9558 -> 0.8304 | -0.1254 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 |
| `multi_04` | multi_part | detailed | 6 -> 10 | 1.00 -> 1.00 | 1.0000 -> 0.7654 | -0.2346 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 |
| `multi_06` | multi_part | detailed | 6 -> 10 | 1.00 -> 1.00 | 0.8503 -> 0.6508 | -0.1995 | 1.0000 -> 1.0000 | 1.0000 -> 1.0000 |
| `hop_06` | multi_hop | detailed | 6 -> 10 | 1.00 -> 1.00 | 0.8319 -> 1.0000 | +0.1681 | 0.1091 -> 0.1091 | 0.0000 -> 0.0000 |
| `prem_10` | false_premise | detailed | 6 -> 10 | 1.00 -> 1.00 | 0.9060 -> 0.8855 | -0.0206 | 0.5714 -> 0.5714 | 0.0000 -> 0.0000 |

---

## 7. Real-Model Validation Readiness (`--mode neural`)

| Component | Target Model | Availability Status | Reason |
| :--- | :--- | :---: | :--- |
| **Neural Embedding** | `BAAI/bge-small-en-v1.5` | **NOT_EXECUTED** | Genuine weights unavailable locally (OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files. Check your internet connection or see how to run ) |
| **Neural Cross-Encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **NOT_EXECUTED** | Genuine weights unavailable locally (OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files. Check your internet connection or see how to run ) |
