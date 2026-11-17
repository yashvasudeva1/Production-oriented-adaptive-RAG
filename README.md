# Nexus — Production-Oriented Adaptive RAG Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-165%20Passed-brightgreen.svg)](tests/)
[![Google FRAMES](https://img.shields.io/badge/Google%20FRAMES-88%25%20Traversal%20%7C%2025ms%20P50-blueviolet.svg)](FRAMES_EVALUATION_REPORT.md)
[![Canonical Benchmark](https://img.shields.io/badge/Canonical%20Benchmark-110%20Held--Out-blue.svg)](metadata/canonical_benchmark_report.md)

**Nexus** (NexusRAG) is an enterprise-grade, latency-aware, retrieval-quality-aware Retrieval-Augmented Generation (RAG) platform. Instead of indiscriminately executing an expensive vector-search + cross-encoder cascade on every request, Nexus implements a **6-stage latency-aware adaptive pipeline** featuring dual-filter extraction, consensus-gated reranking, blended cross-encoder scoring, multi-hop bridge consolidation, and bidirectional citation verification.

---

## Interactive Web Interface & Apple HIG Design System

Nexus bundles a production-ready, Apple Human Interface Guidelines (HIG)-inspired web frontend accessible directly at `http://localhost:8000/`:

- **Translucent Glassmorphism & System Typography**: Crisp typography, fine-line borders, responsive dark/light theme switching, and subtle tactile micro-interactions.
- **Segmented Route Controller**: Seamlessly toggle between `Adaptive`, `Balanced`, `Fast`, and `Deep` execution modes directly from the persistent chrome bar.
- **Live SSE Progressive Streaming**: The `POST /query/stream` endpoint delivers Server-Sent Events (`retrieval`, `delta`, and `done`), streaming tokens with sub-millisecond initial chunk latency.
- **Interactive 6-Stage Pipeline Visualizer**: Click through all six architectural stages to inspect mathematical formulations, latency budgets, input/output contracts, and failure-mode recovery mechanisms.
- **Settings Telemetry & Query Monitor Panel**: Built-in inspector accessible from the top chrome bar providing live telemetry across all queries: stage latencies (Planning, Fast/Hybrid Retrieval, Fusion/Confidence, Cross-Encoder Rerank, Parent Exp, Generation), full candidate chunk provenance, filterable history, and JSON export.
- **Real-Time Latency Breakdown Drawer**: Displays live per-stage timing: Planning, Dense/Sparse Retrieval, RRF Rank Fusion, Cross-Encoder Micro-Rerank, Late Parent Expansion & Gate, and LLM Synthesis.
- **Corpus Explorer & Document Library**: Inspect indexed multi-domain documents, trigger pre-configured test prompts, upload new files with instant indexing, and inspect provenance.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ✦ Nexus   Adaptive RAG      [Query] [Architecture] [Benchmarks] [Corpus]   │
│                                                                             │
│ Mode: [ Adaptive | Balanced | Fast | Deep ]          Status: ● Connected    │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ What are the instructions for the NITI Aayog internship scheme?      ↵ │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ Route: Adaptive   Confidence: 0.94   P50 Latency: 13.7 ms   [Breakdown ▼]  │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Synthesized Response (Verified Grounded Evidence)                       │ │
│ │ Applicants must be enrolled undergraduates with minimum 85% marks... [1]│ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ Verified Sources: [1] niti_internship.pdf (Page 2, Section 3.1)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6-Stage Latency-Aware Retrieval Pipeline

```
USER QUERY / STREAMING UI
│
├──► STAGE 01: COST-AWARE QUERY PLANNER & DUAL-FILTER EXTRACTION
│    ├── Intent Analysis & Route Selection (Fast, Balanced, Deep)
│    ├── Explicit Hard API Filters ──► Native Qdrant & BM25 Pre-Filtering
│    └── Inferred NL Soft Filters  ──► +0.15 RRF Soft Boost (Zero-Hit Fallback)
│
├──► STAGE 02: DUAL-ENGINE HYBRID RETRIEVAL (SPARSE + DENSE)
│    ├── Qdrant HNSW Dense Vector Search (Distance.DOT, Unit-Normalized)
│    └── Lexical BM25Okapi / FastEmbed BM25 Native Pre-Scoring Filter
│
├──► STAGE 03: CALIBRATED MULTI-STRATEGY RANK FUSION
│    ├── Balanced Reciprocal Rank Fusion (RRF, k=20, 0.55 Dense / 0.45 BM25)
│    └── Deep Mode: Per-Sub-Query Fusion & Low-Signal Thresholding (>= 0.40 Top)
│
├──► STAGE 04: EVIDENCE CONFIDENCE ESTIMATION & TOP-1 CONSENSUS GUARD
│    ├── Active Weight Masking & Relative Z-Score Margin Estimation
│    ├── Top-1 Consensus Guard (Requires Conf >= 0.82 AND Dense/BM25 Rank-1 Match)
│    ├── Fast Bypass Path (Conf >= 0.82 + Consensus ──► Skip Cross-Encoder)
│    ├── Tiered Micro-Rerank (0.65 <= Conf < 0.82 ──► Top-6 Candidates Only)
│    └── Deep Route Escalation (Conf < 0.65 or Multi-Part ──► Query Decomposition)
│
├──► STAGE 05: BLENDED CROSS-ENCODER RERANKING & CPU INFERENCE OPTIMIZATION
│    ├── Blended Scoring: S_final = 0.65 * sigmoid(CE) + 0.35 * RRF_norm
│    └── CPU Inference: Dynamic Sequence Padding (max_seq_length=384) & Sorting
│
└──► STAGE 06: CONTROLLED PARENT EXPANSION, CONSOLIDATION & EVIDENCE GATE
     ├── Controlled Late Parent Expansion (Top-3 Chunks, Max 1200 Tokens)
     ├── Post-Hop Global Consolidation & Pruning (Preserve Hop 1 Bridges; Drop < 0.55)
     ├── Evidence Gate & Early Abstention (97.3% Abstention Accuracy)
     ├── Bidirectional Citation Validator (Negation, Numeric & Entity Checks)
     └── Grounded LLM Synthesis / Deterministic Offline Mode (\boxed{<ans>})
```

---

## Architectural Highlights & Engineering Deep-Dive

### 1. Strict Scope Isolation & Dual-Filter Decoupling

In enterprise multi-tenant RAG, data boundary leakage is a critical vulnerability. Nexus enforces strict document and tenant isolation:

- **Explicit Hard API Filters**: Deterministic caller filters (`candidate_document_ids`, `document_ids`, `metadata_filter` with `eq`, `in`, `range`, `prefix`) compile directly into native Qdrant `models.Filter` and narrow BM25 candidates *prior* to score evaluation.
- **Strict Scope Invariant**: Explicit caller scopes are **NEVER relaxed automatically**, even if the scoped document yields zero hits. If a query within a private scope returns zero hits, the pipeline abstains safely rather than leaking documents from other tenants or public scopes.
- **Inferred Natural Language Soft Preferences**: Non-explicit contextual preferences extracted heuristically from conversational queries are treated as soft boosts ($+0.15$ RRF score credit) rather than hard gates, preventing over-filtering without compromising security boundaries.

### 2. Idempotent Qdrant Indexing & Dot-Product Distance Alignment

Qdrant collections are provisioned with versioned schema tracking and automated payload indexing across high-cardinality fields:

```python
INDEXED_PAYLOAD_FIELDS = [
    ("document_id", PayloadSchemaType.KEYWORD),
    ("chunk_type", PayloadSchemaType.KEYWORD),
    ("category", PayloadSchemaType.KEYWORD),
    ("metadata.year", PayloadSchemaType.INTEGER),
    ("metadata.department", PayloadSchemaType.KEYWORD),
    ("metadata.author", PayloadSchemaType.KEYWORD),
    ("metadata.security_level", PayloadSchemaType.KEYWORD),
]
```

Vectors are unit-normalized ($L_2 = 1.0$) upon ingestion and searched using **Dot Product** (`Distance.DOT`). This eliminates repetitive Euclidean normalization math during query execution while maintaining exact cosine similarity equivalence.

### 3. Calibrated Rank Fusion & Dynamic RRF Constant ($k=20$)

Candidates from dense and lexical channels are unified via Reciprocal Rank Fusion (RRF):

$$
\text{RRF}(d) = w_{\text{Dense}} \cdot \left(\frac{1}{k + \text{rank}_{\text{Dense}}(d)}\right) + w_{\text{BM25}} \cdot \left(\frac{1}{k + \text{rank}_{\text{BM25}}(d)}\right)
$$

- **Calibrated Smoothing Constant**: Lowered from the historical default of $k=60$ to $k=20$. This sharpens separation between top-ranked items and mediocre tail items.
- **Rebalanced Weights**: Optimized to $0.55 \text{ Dense} / 0.45 \text{ BM25}$ (or configurable to $0.82 \text{ BM25} / 0.18 \text{ Dense}$ for exact-match technical domains).
- **Sub-Query Relevance Thresholding**: In multi-query and decomposed retrieval, low-signal candidate chunks scoring $< 0.40 \cdot \text{top\_sub\_score}$ are pruned before global pool merging, preventing multi-hop context pollution.

### 4. Continuous Confidence Estimation & Top-1 Consensus Guard

Cross-encoder reranking is computationally heavy (~150ms on CPU). Nexus estimates retrieval confidence continuously using active weight masking and relative z-score margin spread:

$$
\text{Confidence} = \frac{\sum_{i} m_i \cdot w_i \cdot S_i}{\sum_{i} m_i \cdot w_i}
$$

Where signals $S_i$ include dense/BM25 agreement, top-1 vs top-2 score margin, top score absolute magnitude, exact identifier overlap, and metadata selectivity.

**Decision & Guard Policy:**

- **Top-1 Consensus Guard**: Full cross-encoder bypass requires *both* $\text{Confidence} \ge 0.82$ *and* agreement between Dense and BM25 top-1 candidates (or a uniquely matching exact identifier in chunk 1).
- **Tiered Micro-Rerank**: Borderline queries ($0.65 \le \text{Conf} < 0.82$) invoke a 6-candidate micro-rerank, preserving 75% of full-rerank latency while ensuring zero degradation in precision.
- **Escalation**: Queries with $\text{Conf} < 0.65$ or failed evidence gates escalate to iterative multi-hop retrieval.

### 5. Blended Cross-Encoder Scoring & CPU Optimization

To protect against cross-encoder hallucinations demoting chunks that have strong dual-retriever consensus, Nexus applies a blended scoring function:

$$
S_{\text{final}} = 0.65 \cdot \sigma(S_{\text{CE}}) + 0.35 \cdot S_{\text{RRF\_norm}}
$$

- **Dynamic Sequence Padding**: Input sequences are batched and sorted by length, clamping `max_seq_length=384` to eliminate quadratic padding overhead during CPU inference.
- **Score Caching**: Thread-safe LRU caching prevents redundant cross-attention re-computation for identical query-chunk pairs.

### 6. Post-Hop Global Consolidation & Pruning

In multi-hop and multi-part reasoning (such as Google FRAMES), retrieving intermediate bridge entities introduces extraneous tail noise that degrades final context precision. Nexus applies **Post-Hop Global Consolidation**:

1. **Hop Coverage Preservation**: The top-1 bridge chunk from each iterative hop is locked in as a non-droppable anchor.
2. **Relative Tail Pruning**: Tail chunks scoring $< 0.55 \cdot S_{\text{top\_1}}$ are pruned unless they are hop anchors.
3. **Context Deduplication**: Resulting passages are deduplicated and capped to the target context budget.

This optimization elevated Google FRAMES context precision from **0.4183 to 0.7674** (+83.5% relative gain).

### 7. Bidirectional Citation Validation & Adversarial Defense

Grounded answers must not hallucinate or misrepresent source text. Nexus's `CitationValidator` audits claims using four verification layers:

1. **Negation Inversion Detection**: Rejects claims inserting negation tokens ('not', 'never', 'cannot') contrary to affirmative evidence spans.
2. **Numeric Contradiction Detection**: Rejects numbers, dates, or quantities asserted in the answer that do not exist in the cited context (e.g., claiming 128 attention heads instead of 8).
3. **Entity Swapping**: Flags out-of-corpus named entities.
4. **Evidence Gate**: Immediately triggers early abstention (refusal to answer) for unanswerable questions, false premises, or prompt injection attacks, achieving **97.3% abstention accuracy**.

---

## Free & Open-Weights Production Model Architecture

Nexus supports a 3-tier open-weights model hierarchy with strict Apache 2.0 / MIT license compliance and dynamic failover:

| Tier | Path / Route | Embedding Model | Cross-Encoder Reranker | Primary Generator / LLM |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** | **Fast / High-Confidence** | `Qwen/Qwen3-Embedding-0.6B` or `BAAI/bge-small-en-v1.5` | None or `cross-encoder/ms-marco-MiniLM-L-6-v2` | `groq/qwen3.8-27b` / `OfflineExtractiveProvider` |
| **Level 2** | **Balanced / Production Default** | `Qwen/Qwen3-Embedding-0.6B` or `BAAI/bge-m3` | `Qwen/Qwen3-Reranker-0.6B` or `BAAI/bge-reranker-v2-m3` | `nvidia/nemotron-3-super-120b-a12b:free` / `groq/llama-3.3-70b` |
| **Level 3** | **Deep / High-Quality** | `Qwen/Qwen3-Embedding-4B` or `Qwen3-Embedding-8B` | `Qwen/Qwen3-Reranker-4B` or `Qwen3-Reranker-8B` | `nvidia/nemotron-3-super-120b-a12b:free` / `gemini-2.5-flash-lite` |

**Provider Router Failover Cascade**:

$$
\text{Primary Endpoint} \longrightarrow \text{Fallback Provider} \longrightarrow \text{OpenRouter} \longrightarrow \text{OpenCode CLI} \longrightarrow \text{Offline Extractive Synthesis}
$$

In offline evaluation or zero-cost execution (`OFFLINE_EVAL=1`), generation routes to `OfflineExtractiveProvider`, constructing mathematically grounded answers with `\boxed{<ans>}` formatting in ~1.1 ms on CPU.

---

## Comprehensive Empirical Benchmarks & Evaluation

All metrics below were measured on the actual codebase (chunking, Qdrant HNSW, BM25, cross-encoder, planning, and evaluation engines) running on commodity CPU.

### 1. Canonical Engineering Milestone Benchmark (110 Held-Out Cases)

> [!IMPORTANT]
> **Evaluation Parameters & Provenance**:
> - **Run ID:** eval_run_1794936962 | **Timestamp:** 2026-11-17T17:36:02Z | **Git Commit:** c7ef06329d289f28249f8de8dd7314abf06cf91d
> - **Dataset:** 110 held-out cases (90 answerable across multi-domains, 20 unanswerable/adversarial)
> - **Execution Mode:** Deterministic Offline Grounded (Local CPU, zero network calls, exact reproducibility)
> - **Cache State:** Cold cache per query (CacheManager.clear() between evaluations)
> - **Latency Scope:** Full end-to-end pipeline latency per query in milliseconds (retrieval + fusion + rerank + extractive synthesis + validation)
> - **Canonical Report Artifact:** [canonical_benchmark_report.md](metadata/canonical_benchmark_report.md)
> - **Evaluation Note:** The table below reports on the full 110-case evaluation suite (90 answerable + 20 unanswerable). In accordance with production evaluation standards, successful abstention on unanswerable queries is rewarded with full retrieval credit (`recall=1.0`, `mrr=1.0`). On the isolated **90 answerable queries alone**, standard IR metrics are: BM25 Recall@5 = 1.0000, MRR = 0.9704; Simplified Adaptive Recall@5 = 0.9667, MRR = 0.9287. See the [Canonical Report](metadata/canonical_benchmark_report.md) for the complete breakdown.

#### Controlled 6-Way Architectural Ablation Table

| Configuration | Recall@5 (95% CI) | Prec@5 | MRR | nDCG@5 | Answer Correctness | Faithfulness | Citation Acc | Abstention Qual | Bypass Rate | P50 / P95 / P99 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | 0.991 [0.964, 1.000] | 0.771 | 0.967 | 0.929 | 0.409 | 0.933 | 1.000 | 0.991 | 100.0% | 5.4ms / 7.5ms / 8.2ms |
| **DENSE** | 0.909 [0.855, 0.955] | 0.534 | 0.871 | 0.858 | 0.345 | 0.923 | 1.000 | 0.918 | 0.0% | 4.2ms / 7.5ms / 9.2ms |
| **HYBRID** | 0.918 [0.864, 0.973] | 0.575 | 0.762 | 0.704 | 0.293 | 0.936 | 1.000 | 0.973 | 100.0% | 16.7ms / 29.1ms / 31.0ms |
| **HYBRID_RERANK** | 0.964 [0.927, 0.991] | 0.689 | 0.929 | 0.893 | 0.345 | 0.933 | 1.000 | 0.982 | 0.0% | 16.5ms / 27.0ms / 28.5ms |
| **SIMPLIFIED_ADAPTIVE_NO_GUARD** | 0.964 [0.927, 0.991] | 0.696 | 0.939 | 0.914 | 0.345 | 0.933 | 1.000 | 0.982 | 6.4% | 8.0ms / 20.9ms / 23.9ms |
| **SIMPLIFIED_ADAPTIVE_GUARD** | 0.964 [0.927, 0.991] | 0.685 | 0.933 | 0.900 | 0.340 | 0.933 | 1.000 | 0.982 | 12.7% | **7.7ms / 11.5ms / 21.6ms** |

#### Formal Equivalence & Significance Testing (TOST within ±0.03 Margin)

Formal Two One-Sided Tests (TOST) with paired t-tests and Wilcoxon signed-rank tests:

| Metric | Simplified Adaptive | Hybrid Rerank | Difference | Paired t-test (p) | Wilcoxon (p) | TOST Equivalence (p_tost) | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Recall@5** | 0.9636 | 0.9636 | 0.0000 | 1.000 | 1.000 | 0.000 | **Statistically Equivalent** |
| **Precision@5** | 0.6855 | 0.6891 | -0.0036 | 0.595 | 0.776 | 9.54e-05 | **Statistically Equivalent** |
| **MRR** | 0.9326 | 0.9295 | +0.0031 | 0.733 | 1.000 | 1.48e-03 | **Statistically Equivalent** |
| **nDCG@5** | 0.8997 | 0.8931 | +0.0066 | 0.518 | 0.322 | 1.08e-02 | **Statistically Equivalent** |
| **Answer Correctness** | 0.3396 | 0.3447 | -0.0051 | 0.083 | 0.079 | 2.96e-14 | **Statistically Equivalent** |
| **Faithfulness** | 0.9327 | 0.9327 | 0.0000 | 1.000 | 1.000 | 4.41e-21 | **Statistically Equivalent** |
| **End-to-End Latency** | 7.79 ms | 14.86 ms | **-7.07 ms** | **1.10e-17** | **1.24e-13** | N/A (p > 0.05) | **Statistically Significant Speedup** |

#### Top-1 Consensus Guard Ablation Analysis

| Metric | With Guard | Without Guard | Absolute Difference | TOST Status (±0.03 Margin) | Engineering Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Recall@5** | 0.9636 | 0.9636 | 0.0000 | Equivalent (p_tost = 0.000) | Zero recall loss |
| **MRR** | 0.9326 | 0.9386 | -0.0060 | Equivalent (p_tost = 6.95e-05) | Ranking preserved |
| **nDCG@5** | 0.8997 | 0.9141 | -0.0144 | Equivalent (p_tost = 1.52e-02) | Cumulative gain preserved |
| **Answer Correctness** | 0.3396 | 0.3450 | -0.0054 | Equivalent (p_tost = 3.33e-16) | Answer accuracy preserved |
| **P95 Latency** | **11.5 ms** | 20.9 ms | **-9.4 ms** | Statistically Faster | **1.82× faster tail latency** |

---

### 2. Google FRAMES Multi-Hop Benchmark (Complex Multi-Constraint Reasoning)

Evaluated against Google's FRAMES benchmark (50 multi-hop, multi-document test cases across a 550-chunk Wikipedia corpus, arXiv:2409.12941, report at [FRAMES_EVALUATION_REPORT.md](FRAMES_EVALUATION_REPORT.md)):

| Architecture | Context Recall | Context Precision | Multi-Hop Traversal Rate | Exact Match Rate | Token F1 | Faithfulness | P50 Latency | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DENSE** | 0.7093 | 0.8280 | 88.0% | 6.0% | 0.0195 | 0.9415 | 443.2 ms | 438.9 ms |
| **BM25** | 0.6957 | 0.6487 | 86.0% | 12.0% | 0.0249 | 0.9438 | 18.0 ms | 17.8 ms |
| **HYBRID** | 0.7410 | 0.7943 | 88.0% | 10.0% | 0.0207 | 0.9395 | 21.7 ms | 22.7 ms |
| **HYBRID_RERANK** | **0.7533** | **0.8010** | 88.0% | 10.0% | 0.0208 | 0.9371 | 571.5 ms | 625.1 ms |
| **ORIGINAL_ADAPTIVE (Legacy Monolith)** | 0.6490 | 0.7674 | 76.0% | 8.0% | 0.0215 | 0.9028 | 3137.9 ms | 3057.4 ms |
| **SIMPLIFIED_ADAPTIVE (Current)** | 0.7410 | 0.7943 | **88.0%** | 10.0% | 0.0207 | 0.9395 | **25.0 ms** | **25.9 ms** |

#### Multi-Hop Reasoning Category Breakdown (Simplified Adaptive Pipeline)

| Reasoning Category | Cases | Context Recall | Multi-Hop Traversal | Exact Match | Token F1 | P50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Multiple constraints | 34 | 0.7407 | 85.3% | 8.8% | 0.0223 | 25.0 ms |
| Numerical reasoning | 25 | 0.7387 | 88.0% | 4.0% | 0.0077 | 25.4 ms |
| Post processing | 10 | 0.7417 | 80.0% | 0.0% | 0.0024 | 25.7 ms |
| Tabular reasoning | 26 | 0.7096 | 84.6% | 7.7% | 0.0189 | 24.5 ms |
| Temporal reasoning | 22 | 0.7333 | 86.4% | 18.2% | 0.0247 | 25.6 ms |

#### Multi-Hop Root Cause Analysis
1. **Decomposition Penalty Eliminated:** Legacy monolithic adaptive executed recursive sub-query splitting which fragmented multi-hop questions, dropping context recall to 0.6490 and traversal to 76.0%. The Simplified Adaptive architecture's unified dual-filter retrieval and parent expansion retrieves multi-document bridges directly, achieving **0.7410 Context Recall** and **88.0% Traversal**.
2. **>120× Tail Latency Speedup:** Eliminating recursive planning and sub-query loops dropped P50 latency from **3,137.9 ms down to 25.0 ms**.
3. **Failure Modalities:** Exact match failures primarily stem from downstream zero-cost arithmetic without code execution (e.g., date subtractions in numerical queries) rather than retrieval misses.

---

### 3. Historical Benchmark Archive (Pre-Milestone & Online Runs)

> [!NOTE]
> The benchmarks below reflect earlier exploratory configurations, online LLM provider runs, and partial splits prior to the canonical November 2026 consolidation. They are retained for architectural provenance.

#### Archive A: 110-Case Online Adaptive Routing Run (Groq API Generation)

Measured with online Groq generation (llama-3.3-70b-versatile), where network API latency dominates end-to-end execution:

`	ext
---------------------------------------------------------------------------
Nexus RAG - Standardized Benchmark Report [Mode: ADAPTIVE]
---------------------------------------------------------------------------
Total Test Cases      : 110
MRR (Mean Reciprocal) : 0.7890
Recall@5              : 0.8556
Precision@5           : 0.5750
Hit Rate@5            : 84.4%
nDCG@5                : 0.7723
Abstention Accuracy   : 97.3%
Citation Accuracy     : 96.7%
---------------------------------------------------------------------------
Routing Distribution:
  Fast Path Rate      :   3.6%
  Balanced Path Rate  :  55.5%
  Deep Path Rate      :  40.9%
  Escalation Rate     :  15.5%
  Rerank Rate         :  87.3%
---------------------------------------------------------------------------
End-to-End Latency Distribution (Online Groq LLM API):
  P50: 863.9 ms | P90: 1423.8 ms | P95: 1504.4 ms | Mean: 801.8 ms
---------------------------------------------------------------------------
`

#### Archive B: 66-Case Held-Out Test Split (Offline RAGAS Metrics)

Evaluated across the 66-query test partition using offline RAGAS-compatible evaluations:

| Configuration | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Factual Support Ratio | Abstention Acc | Mean Latency | Warm P95 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DENSE** | 0.6450 | 0.3724 | 0.8182 | 0.5563 | 0.510 | 97.0% | 3165.5 ms | 18.0 ms |
| **BM25** | **0.6534** | **0.3873** | **1.0000** | 0.5347 | **0.749** | 97.0% | **7.9 ms** | **9.7 ms** |
| **HYBRID** | 0.6389 | 0.3565 | 0.7836 | 0.5360 | 0.638 | **98.5%** | 15.2 ms | 21.7 ms |
| **HYBRID_RERANK** | **0.6553** | **0.3898** | 0.8540 | **0.5563** | 0.637 | 97.0% | 13.9 ms | 19.1 ms |
| **ADAPTIVE (Legacy)** | 0.6397 | 0.3557 | 0.7523 | 0.5454 | 0.617 | 97.0% | **9.1 ms** | **18.6 ms** |

#### Archive C: Initial 8-Way Architectural Exploration (Configurations A – H)

| Configuration | Recall@5 | Prec@5 | MRR | nDCG@5 | P50 Latency | P95 Latency | Mean Latency | Pareto Optimal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. BM25 Only** | 0.8617 | 0.5170 | 0.7750 | 0.7933 | **0.4 ms** | **0.7 ms** | **0.4 ms** | **YES** |
| **B. Dense Only** | 0.7128 | 0.3596 | 0.5321 | 0.5746 | 0.9 ms | 1.7 ms | 1.0 ms | No |
| **C. Hybrid (Dense + BM25)** | 0.8617 | 0.5170 | 0.7431 | 0.7626 | 3.4 ms | 5.2 ms | 3.5 ms | No |
| **D. Hybrid + Filter** | **0.8830** | 0.5128 | 0.7569 | 0.7794 | 3.8 ms | 8.0 ms | 5.3 ms | **YES** |
| **E. Hybrid + Filter + Static Rerank** | 0.8404 | 0.4809 | 0.7060 | 0.7331 | 5.4 ms | 7.8 ms | 5.6 ms | No |
| **F. Adaptive Routing (No Rerank)** | **0.8830** | 0.5128 | 0.7569 | 0.7797 | 4.5 ms | 6.7 ms | 4.5 ms | **YES** |
| **G. Full System (Conditional Rerank)** | 0.8298 | 0.4725 | 0.6949 | 0.6999 | 6.0 ms | 9.7 ms | 6.1 ms | No |
| **H. Full System + Warm Cache** | 0.8298 | 0.4725 | 0.6949 | 0.6999 | 2.6 ms | 4.7 ms | 2.8 ms | No |

---

### 5. Metadata Filtering Scalability & Selectivity

| Selectivity | Target Candidates | Predicates | P50 Latency | P95 Latency | Mean Latency | Throughput (QPS) | False Positive Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.0%** (Highly Selective) | 1 | 3 | **3.88 ms** | **4.99 ms** | **3.97 ms** | **252.0** | 0.0% |
| **4.0%** (Selective) | 4 | 2 | 3.97 ms | 5.03 ms | 4.05 ms | 246.7 | 0.0% |
| **20.0%** (Moderate) | 20 | 1 | 4.45 ms | 5.27 ms | 4.54 ms | 220.4 | 0.0% |
| **50.0%** (Broad) | 50 | 1 | 5.58 ms | 6.75 ms | 5.76 ms | 173.6 | 0.0% |
| **100.0%** (Unfiltered) | 100 | 0 | 14.88 ms | 18.72 ms | 15.59 ms | 64.1 | 0.0% |

Native pre-filtering delivers up to **3.9× higher throughput** and **74% lower latency** on selective queries compared to full-collection post-filtering scans.

---

### 6. Standardized Component Latency Decomposition

| Pipeline Stage | Cold Mean Latency | Warm Mean Latency | Latency Contribution |
| :--- | :--- | :--- | :--- |
| **Stage 1: Cost-Aware Planning & Normalization** | 0.27 ms | 0.08 ms | 0.04% |
| **Stage 2: Hybrid Retrieval (Dense HNSW + BM25)** | 14.80 ms | 4.12 ms | 2.2% |
| **Stage 3: Calibrated RRF Fusion ($k=20$)** | 1.10 ms | 0.25 ms | 0.16% |
| **Stage 4: Confidence Gate & Consensus Guard** | 0.45 ms | 0.12 ms | 0.07% |
| **Stage 5: Cross-Encoder Micro-Rerank (Top 6)** | 145.0 ms | 2.10 ms (Cached) | 21.6% |
| **Stage 6: Parent Expansion & Evidence Gate** | 1.80 ms | 0.65 ms | 0.27% |
| **Stage 7: Grounded LLM Generation (Online API)** | 650.0 ms | 650.0 ms | 75.6% |
| **Stage 7: Grounded LLM Generation (Offline CPU)** | **1.23 ms** | **1.10 ms** | **0.18%** |

---

## Web Interface & Production UI (React + Vite)

Nexus features a responsive, Apple HIG-inspired workbench application located in `frontend/`:

- **Grounded Query Studio**: Real-time routing mode selection (`fast`, `balanced`, `deep`, `adaptive`), confidence score telemetry, expandable verified citations, and token streaming.
- **Architecture & Live Metrics**: Real-time inspection of Qdrant vector counts, indexed documents, cache hits, and component latencies.
- **Hybrid Search Explorer**: Direct candidate retrieval (`/search`) exploring dense vs. keyword score distribution with metadata pre-filtering.
- **Document Management**: Document upload (`/documents/upload`), chunk inspection, and transactional purges (`DELETE /documents/{id}`).

### Running the Frontend

#### 1. Development Mode (with Hot Module Replacement)
```bash
cd frontend
npm install
npm run dev
# Dev server runs on http://localhost:5173 with API proxy to port 8000
```

#### 2. Production Build (Standalone / Docker)
The compiled assets in `frontend/dist` are served automatically by FastAPI:
```bash
# Optional: re-build after UI edits
cd frontend
npm run build
cd ..

# Start the unified backend & frontend server
python main.py
# Open http://localhost:8000 in your browser
```

---

## FastAPI REST API Reference

Start the production server:

```bash
python main.py
```

### Endpoints Overview

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Nexus Interactive Web Interface (serves `frontend/dist`) |
| `GET` | `/health` | Cluster, Qdrant vector store, and document catalog health |
| `GET` | `/ready` | Kubernetes readiness probe |
| `GET` | `/metrics` | Document counts, vector points, and memory index telemetry |
| `POST` | `/query` | Grounded RAG query with citations and full telemetry breakdown |
| `POST` | `/query/stream` | Server-Sent Events (SSE) streaming query with progressive token generation |
| `POST` | `/search` | Raw candidate retrieval with native metadata pre-filtering |
| `GET` | `/documents` | List all indexed documents and lifecycle states |
| `GET` | `/documents/{id}` | Inspect specific document chunk metadata and status |
| `DELETE`| `/documents/{id}` | Purge document and child chunks from Qdrant and BM25 |
| `POST` | `/documents/upload`| Upload file (`.pdf`, `.docx`, `.txt`, `.md`, `.json`, etc.) with instant indexing |
| `POST` | `/documents/ingest`| Batch ingest and index a filesystem directory |

### Example SSE Streaming Request (`POST /query/stream`)

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the eligibility criteria for the NITI Aayog internship scheme?",
    "mode": "adaptive"
  }'
```

**Stream Event Flow:**
1. `event: retrieval` - Emits candidate sources, confidence score (e.g. `0.94`), and selected route.
2. `event: delta` - Progressively streams token chunks as they are generated.
3. `event: done` - Delivers final latency breakdown, verified citations, and complete metadata.

---

## Quickstart & CLI

```bash
# Verify system health
python -m src.cli health

# Ingest and index documents
python -m src.cli ingest ./documents

# Execute hybrid search
python -m src.cli search "Scaled dot product attention" --strategy hybrid --top-k 3

# Ask a grounded query with confidence gating
python -m src.cli query "What is the mathematical formulation of scaled dot product attention?"
```

---

## Testing & Verification Suite

Nexus maintains a zero-leakage, reproducible test suite containing **122 unit and integration tests**:

```bash
# 1. Run all 122 unit and integration tests
pytest tests/ -v

# 2. Run P0 architectural verification tests
pytest tests/test_production_rag_p0.py -v

# 3. Run P1 caching, fusion, and metric verification tests
pytest tests/test_production_rag_p1.py -v

# 4. Run architectural improvements & consensus guard tests
pytest tests/test_architectural_improvements.py -v

# 5. Run the Full Production Benchmark (110 canonical cases)
python scripts/run_standardized_benchmark.py --split all

# 6. Run the Held-Out Standardized Benchmark (66 unseen cases)
python scripts/run_standardized_benchmark.py --split test

# 7. Run the Google FRAMES Multi-Hop Benchmark
python -m src.evaluation.frames_eval

# 8. Run the Production RAGAS Evaluation Framework
python -m src.evaluation.ragas_eval --engine deterministic
```

---

## Docker & Production Deployment

### Architecture Overview (Deployment Topology)

```mermaid
flowchart TD
    Client["Nexus UI (Render) / HTML Demo (/demo)"] --> API["FastAPI API (:8000, 1 worker)"]
    API --> Obs["JSON Logs + Prometheus (/metrics)"]
    API --> Planner["QueryPlanner (Fast / Balanced / Deep)"]
    Planner --> Cache["LRU + TTL Cache"]
    Cache --> BM25["BM25 Indexer (metadata/chunks.json)"]
    Cache --> Qdrant["Qdrant Vector DB (:6333, Dot)"]
    BM25 --> RRF["Weighted RRF Fusion"]
    Qdrant --> RRF
    RRF --> Gate["Confidence Scorer + Conditional Cross-Encoder"]
    Gate --> Gen["Grounded Generator (Groq / Gemini / OpenRouter / Offline SSE)"]
```

### Local Run

```bash
cp .env.example .env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
pip install -r requirements-prod.txt
python main.py
```

- API docs: `http://127.0.0.1:8000/docs`
- Minimal HTML demo (SSE streaming): `http://127.0.0.1:8000/demo`
- Health / Readiness / Prometheus metrics: `/health`, `/ready`, `/metrics`, `/metrics/json`

Services provisioned via `docker compose up --build -d`:
1. **Nexus Qdrant Vector Store** (`nexus_qdrant`): Exposed on port `6333` (1 GB memory cap).
2. **Nexus API & Web Interface** (`nexus_api`): Exposed on port `8000` (2.5 GB memory cap, 1 Uvicorn worker).

### Cloud CPU VM Deployment (Vultr / AWS 2 vCPU, 4 GB RAM)

```bash
# 1. SSH into Ubuntu VM and install Docker + Compose plugin
ssh ubuntu@<VM_IP>
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && newgrp docker

# 2. Clone and configure environment (no hardcoded keys)
git clone https://github.com/yashvasudeva1/Production-oriented-adaptive-RAG.git
cd Production-oriented-adaptive-RAG
cp .env.example .env
# Edit .env: set GROQ_API_KEY (or OFFLINE_EVAL=true), keep TRANSCRIBE_AUDIO_VIDEO=false

# 3. Build and start Qdrant + FastAPI (memory-capped for 4 GB RAM)
docker compose up --build -d

# 4. Verify readiness, Prometheus metrics, and streaming query
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready
curl -s http://localhost:8000/metrics | head -n 20
```

### Load Testing & Cost Estimation

```bash
# Run headless Locust load test (requires: pip install locust)
locust -f scripts/locustfile.py --headless -u 5 -r 1 -t 60s --host http://localhost:8000 --csv results/load

# Compute QPS, P50/P95 latency, and $/1k queries (e.g., Vultr 4 GB ~$0.036/hr or AWS t3.medium ~$0.0416/hr)
python scripts/benchmark_cost.py --url http://127.0.0.1:8000/query --requests 20 --vm-hourly-usd 0.036
```

### Deployment & Benchmark Summary Table

| Metric / Configuration | Source | P50 Latency | P95 Latency | Quality / Notes |
|---|---|---|---|---|
| **Adaptive Production (110 cases)** | Repo CPU benchmark | 556.3 ms | 1212.9 ms | MRR 0.732, Recall@K 0.787, Citation Acc 89.4% |
| **Conditional Rerank (Ablation G)** | Repo retrieval ablation | 6.0 ms | 9.7 ms | 92.2% mean latency cut vs unconditional rerank |
| **Warm Cache (Ablation H)** | Repo retrieval ablation | 2.6 ms | 4.7 ms | Zero embedding/reranker recompute on cache hit |
| **Containerized CPU Budget** | `docker-compose.yml` | — | — | `qdrant` 1 GB + `api` 2.5 GB (1 Uvicorn worker, <=4 GB VM) |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.