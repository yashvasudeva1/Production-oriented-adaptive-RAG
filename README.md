# Production-oriented Adaptive RAG: Adaptive, Filter-Aware, Hybrid, Confidence-Gated RAG Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-88%20Passed-brightgreen.svg)](tests/)

**Production-oriented Adaptive RAG** is a latency-aware, retrieval-quality-aware Retrieval-Augmented Generation (RAG) platform. Instead of always running an expensive vector-search + cross-encoder cascade, it implements an **adaptive, filter-aware, hybrid, confidence-gated pipeline** designed to balance retrieval quality against latency and compute cost.

### Core Architectural Principle

```
CHEAP RETRIEVAL
──► FILTER-AWARE SEARCH (Qdrant & BM25 Native Pre-Filtering)
──► HYBRID CANDIDATE GENERATION (Dense + Lexical)
──► MULTI-STRATEGY RANK FUSION (Weighted RRF / Score-Normalized)
──► EVIDENCE CONFIDENCE ESTIMATION (Agreement, Margins, Overlap)
──► CONDITIONAL RERANKING (Bypass Cross-Encoder when Confidence >= 0.85)
──► CONTROLLED LATE PARENT EXPANSION (Top-K Chunks with Token Budgets)
──► EVIDENCE GATE (Early Abstention for False Premises)
──► GROUNDED GENERATION & BIDIRECTIONAL CITATION VALIDATION
```

---

## Complete End-to-End Architecture

```
USER QUERY
│
▼
Query Interpretation & Extraction
┌───────────────────┴───────────────────┐
▼                                       ▼
Query Categorization                    Canonical RetrievalFilter
(Cost-Aware Planner:                    • Hard Filters (payload-level)
 FAST / BALANCED / DEEP)                • Soft Preferences (ranking boost)
│                                       • Retrieval Hints (exact phrases)
│                                       │
▼                                       ▼
┌─────────────────────────── Multi-Level Cache ───────────────────────────┐
│ Check Embedding Cache -> Check Retrieval Cache (Key: query+filters+index_v)│
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ (cache miss)
┌───────────────────────────────────┴─────────────────────────────────────┐
▼                                                                         ▼
Dense Retrieval                                                    Keyword Retrieval
(Qdrant Search with                                                (BM25 with Native
 Native Payload Filter)                                             Pre-Scoring Filter)
│                                                                         │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
                          Candidate Rank Fusion
                        (Weighted RRF or Score-Norm)
                                    │
                                    ▼
                    Evidence Confidence Estimation
                    • Dense/BM25 Agreement
                    • Score Margin & Top Score
                    • Exact Identifier Coverage
                    • Metadata Selectivity
                                    │
┌───────────────────────────────────┴─────────────────────────────────────┐
▼                                                                         ▼
Confidence >= 0.85?                                               Confidence < 0.85?
(High Agreement / Score)                                          (Ambiguous / Low Margin)
│                                                                         │
│ [BYPASS RERANKER]                                                       │ [INVOKE CROSS-ENCODER]
│ (Saves significant latency when triggered)                              ▼
│                                                                 Cross-Encoder Reranker
│                                                                 (Query-Doc Pair Cached)
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
                    Controlled Late Parent Expansion
                    (Top-K chunks only; bounded tokens)
                                    │
                                    ▼
                          Context Budget Filter
                                    │
                                    ▼
                              Evidence Gate
┌───────────────────────────────────┴─────────────────────────────────────┐
[PASS]                                                                  [FAIL]
│                                                                         │
▼                                                                         ▼
Grounded LLM Generator                                          Early Abstention Response
│
▼
Bidirectional Citation Validator
│
▼
GROUNDED ANSWER
```

---

## Architectural Highlights & Engineering Deep-Dive

### 1. First-Class Metadata Filtering (`RetrievalFilter`)

Many RAG systems either post-filter retrieved candidates (reducing effective top-k) or enumerate matching document IDs in Python and pass large ID lists into vector queries. This platform introduces `RetrievalFilter` and `QueryConstraints`:

- **Hard Filters**: Deterministic constraints (`eq`, `in`, `range`, `prefix`) compiled into native Qdrant `models.Filter` payload queries.
- **BM25 Native Pre-Filtering**: Lexical search candidate indices are narrowed *before* computing BM25 scores, reducing search latency on selective queries (up to ~75% in measured cases).
- **Soft Preferences & Hints**: Scoring adjustments and exact-term extraction for technical codes and identifiers.

### 2. Idempotent Qdrant Payload Indexing & Vector Distance Alignment

Payload schema indices are created at startup for high-cardinality fields:

- `document_id` (Keyword)
- `chunk_type` (Keyword)
- `category` (Keyword)
- `metadata.year` (Integer)
- `metadata.department` (Keyword)
- `metadata.author` (Keyword)
- `metadata.security_level` (Keyword)
- `metadata.language` (Keyword)

Index creation is idempotent and version-tracked. Vector distance is set to **Dot Product** (`Distance.DOT`) on unit-normalized embeddings to avoid repeated Euclidean normalization overhead.

### 3. Evidence-Based Confidence Estimation & Conditional Reranking

Unconditional cross-encoder reranking is a major latency source in many production RAG systems. This platform estimates retrieval confidence from measurable signals:

$$
\text{Confidence} = w_1 \cdot \text{Agreement}_{\text{dense,bm25}} + w_2 \cdot \text{Margin}_{\text{top1}-\text{top2}} + w_3 \cdot \text{ExactIdMatches} + w_4 \cdot \text{TermCoverage}
$$

**Decision Policy:**

- ≥ 0.85 → **Bypass reranker** (preserves fusion ordering)
- 0.60 ≤ Conf < 0.85 → **Conditional reranking** of top 15–20 candidates
- < 0.60 → **Escalate** to deep route with multi-query decomposition

> **Note on measured behavior**: In the full 110-case Adaptive benchmark, the fast path was taken on only 3.6% of queries and the reranker was invoked on 87.3% of queries. The confidence gate therefore provides latency savings primarily on high-agreement queries; on the broader evaluation set the majority of traffic still incurs reranking cost.

### 4. Controlled Late Parent Expansion

Parent-child chunking indexes smaller child passages (150–300 tokens) for retrieval precision. Expansion is applied only to the top-ranked chunks with hard limits:

- `max_parent_expansions = 3`
- `max_parent_tokens = 1200`

### 5. Multi-Level Caching Subsystem

Thread-safe LRU + TTL cache (`LRUTTLCache`):

- **Query Embedding Cache**
- **Filtered Retrieval Cache** (keyed on normalized query + filters + index version)
- **Cross-Encoder Score Cache**

Cache is invalidated via `bump_index_version()` on index changes.

### 6. Multi-Strategy Rank Fusion

Configurable via `FUSION_METHOD`:

- **Weighted Reciprocal Rank Fusion (RRF)** (default):

$$
\text{RRF}(d) = 0.82 \cdot \left(\frac{1}{60 + \text{rank}_{\text{BM25}}(d) + 1}\right) + 0.18 \cdot \left(\frac{1}{60 + \text{rank}_{\text{Dense}}(d) + 1}\right)
$$

- **Score-Normalized Linear Fusion**

---

## Default Models Configuration

- **Dense Embedding Model:** `thenlper/gte-small` (384-dim, unit-normalized, 512-token context)
- **Vector Distance:** `Distance.DOT`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional `BAAI/bge-reranker-large`)
- **Fusion:** Weighted RRF (`w_BM25 = 0.82`, `w_Dense = 0.18`)
- **Chunking:** ~200 tokens with 30-token overlap

---

## Benchmark & Empirical Evaluation

All numbers below were measured on the actual codebase (chunking, Qdrant, BM25, cross-encoder, planning) running on commodity CPU. Latencies are end-to-end unless otherwise noted.

### 1. Full Production Benchmark Report (110 Cases) — Mode: ADAPTIVE

```
---------------------------------------------------------------------------
Production-oriented Adaptive RAG — Benchmark Evaluation Report [Mode: ADAPTIVE]
---------------------------------------------------------------------------
Total Test Cases      : 110
MRR (Mean Reciprocal) : 0.7890
Recall@K              : 0.8556
Precision@K           : 0.5750
Hit Rate@K            : 84.4%
nDCG@K                : 0.7723
Abstention Accuracy   : 97.3%
Citation Accuracy     : 96.7%
---------------------------------------------------------------------------
Routing Distribution:
  Fast Path Rate      :   3.6%
  Balanced Path Rate  :  55.5%
  Deep Path Rate      :  40.9%
  Escalation Rate     :  15.5%
  Rerank Rate         :  87.3%
  Multi-Query Rate    :  28.2%
  Parent Expansion    :  56.4%
  Abstention Rate     :  20.9%
---------------------------------------------------------------------------
Latency Distribution (Full Execution on Commodity CPU):
  Min   :    2.0 ms
  P50   :  863.9 ms
  P90   : 1423.8 ms
  P95   : 1504.4 ms
  P99   : 1851.0 ms
  Mean  :  801.8 ms
  Max   : 3179.5 ms
---------------------------------------------------------------------------
```

**Per-Category Results**

| Category            | Cases | Recall | Prec  | MRR   | Citations | P50 (ms) | P95 (ms) |
|---------------------|-------|--------|-------|-------|-----------|----------|----------|
| exact               | 10    | 0.80   | 0.43  | 0.73  | 100.0%    | 122.5    | 2189.9   |
| conceptual          | 10    | 1.00   | 0.62  | 1.00  | 100.0%    | 657.6    | 938.3    |
| procedural          | 10    | 0.90   | 0.40  | 0.69  | 100.0%    | 774.8    | 1008.5   |
| comparison          | 10    | 0.90   | 0.71  | 0.85  | 100.0%    | 1404.2   | 1789.3   |
| summarization       | 10    | 0.70   | 0.45  | 0.61  | 90.0%     | 1143.9   | 1411.5   |
| multi_part          | 10    | 0.80   | 0.50  | 0.80  | 90.0%     | 1214.4   | 1496.9   |
| metadata_filtered   | 10    | 0.90   | 0.85  | 0.90  | 90.0%     | 194.8    | 905.3    |
| multi_hop           | 10    | 1.00   | 0.78  | 0.87  | 100.0%    | 841.2    | 1503.7   |
| unanswerable        | 10    | N/A    | N/A   | N/A   | N/A       | 977.4    | 1135.1   |
| false_premise       | 10    | 0.70   | 0.44  | 0.65  | 100.0%    | 548.4    | 1155.3   |
| adversarial         | 10    | N/A    | N/A   | N/A   | N/A       | 960.5    | 1138.4   |

### 2. Architectural Ablation Study

| Configuration                           | Recall@5 | Prec@5 | MRR    | nDCG@5 | P50 Latency | P95 Latency | Mean Latency | Pareto Optimal |
|-----------------------------------------|----------|--------|--------|--------|-------------|-------------|--------------|----------------|
| **A. BM25 Only**                        | 0.8617   | 0.5170 | 0.7750 | 0.7933 | 0.4 ms      | 0.7 ms      | 0.4 ms       | **YES**        |
| **B. Dense Only**                       | 0.7128   | 0.3596 | 0.5321 | 0.5746 | 0.9 ms      | 1.7 ms      | 1.0 ms       | No             |
| **C. Hybrid (Dense + BM25)**            | 0.8617   | 0.5170 | 0.7431 | 0.7626 | 3.4 ms      | 5.2 ms      | 3.5 ms       | No             |
| **D. Hybrid + Filter**                  | **0.8830** | 0.5128 | 0.7569 | 0.7794 | 3.8 ms      | 8.0 ms      | 5.3 ms       | **YES**        |
| **E. Hybrid + Filter + Static Rerank**  | 0.8404   | 0.4809 | 0.7060 | 0.7331 | 5.4 ms      | 7.8 ms      | 5.6 ms       | No             |
| **F. Adaptive Routing (No Rerank)**     | **0.8830** | 0.5128 | 0.7569 | 0.7797 | 4.5 ms      | 6.7 ms      | 4.5 ms       | **YES**        |
| **G. Full System (Conditional Rerank)** | 0.8298   | 0.4725 | 0.6949 | 0.6999 | 6.0 ms      | 9.7 ms      | 6.1 ms       | No             |
| **H. Full System + Warm Cache**         | 0.8298   | 0.4725 | 0.6949 | 0.6999 | 2.6 ms      | 4.7 ms      | 2.8 ms       | No             |

> **Important note on ablation vs full benchmark**: The ablation study reports much lower latencies (single-digit to low double-digit milliseconds) than the full 110-case Adaptive run (P50 864 ms). Differences arise from subset size, cache state, inclusion of generation + citation validation, and routing behavior. The ablation is useful for relative component impact; the full benchmark better reflects end-to-end Adaptive mode on the complete test suite.

### 3. Metadata Filtering Scalability

| Selectivity                      | Target Candidates | Predicates | P50 Latency | P95 Latency | Mean Latency | Throughput (QPS) | False Positive Rate |
|----------------------------------|-------------------|------------|-------------|-------------|--------------|------------------|---------------------|
| **1.0%** (Highly Selective)      | 1                 | 3          | 3.88 ms     | 4.99 ms     | 3.97 ms      | 252.0            | 0.0%                |
| **4.0%** (Selective)             | 4                 | 2          | 3.97 ms     | 5.03 ms     | 4.05 ms      | 246.7            | 0.0%                |
| **20.0%** (Moderate)             | 20                | 1          | 4.45 ms     | 5.27 ms     | 4.54 ms      | 220.4            | 0.0%                |
| **50.0%** (Broad)                | 50                | 1          | 5.58 ms     | 6.75 ms     | 5.76 ms      | 173.6            | 0.0%                |
| **100.0%** (Unfiltered)          | 100               | 0          | 14.88 ms    | 18.72 ms    | 15.59 ms     | 64.1             | 0.0%                |

Pre-filtering yields up to ~3.9× higher throughput on selective queries compared with unfiltered scans.

### 4. Comparative Configuration Matrix (110 Cases)

| Configuration              | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Answer Correctness (Token F1) | Abstention Acc | Mean Latency |
|----------------------------|-------------------|----------------|--------------|------------------|-------------------------------|----------------|--------------|
| **DENSE**                  | 0.5937            | 0.3569         | 0.8561       | 0.5352           | 0.1625                        | 97.3%          | 1235.8 ms    |
| **BM25**                   | 0.6445            | 0.3925         | 0.9333       | 0.5503           | 0.1561                        | 98.2%          | 2.7 ms       |
| **HYBRID (0.82/0.18 RRF)** | 0.6182            | 0.3787         | 0.8000       | 0.5330           | 0.1603                        | 98.2%          | 12.4 ms      |
| **HYBRID + RERANK**        | **0.6462**        | **0.4037**     | **0.9773**   | **0.5352**       | 0.1625                        | 97.3%          | 105.2 ms     |
| **ADAPTIVE (Production)**  | 0.6197            | 0.3810         | 0.7909       | 0.5357           | **0.1634**                    | 97.3%          | 3.0 ms       |

> Token-F1 Answer Correctness is reported for completeness but is sensitive to response length. When an LLM judge is used, semantic correctness on answerable queries typically falls in the 0.95–0.99 range (see RAGAS notes below).

### 5. RAGAS LLM-Judge Notes

- Surface token-F1 understates semantic correctness because the system produces longer, citation-bearing answers.
- Example: `exact_01` → LLM-judge Correctness 0.9551 / Faithfulness 1.0000 (token-F1 0.6667).
- Rate-limit mitigation for Groq free tier: `max_tokens=850`, `max_workers=1`.

### 6. Operating Modes

- **`mode="fast"`** — Pure BM25 + native pre-filtering. Lowest latency, strongest on keyword-heavy queries.
- **`mode="adaptive"`** (default) — Confidence-gated hybrid retrieval. Bypasses the cross-encoder when confidence ≥ 0.85.
- **`mode="hybrid_rerank"`** — Always reranks; prioritizes maximum retrieval quality over latency.

---

## Quickstart & CLI

```bash
python -m src.cli health
python -m src.cli ingest ./documents
python -m src.cli search "Transformer attention mechanism" --strategy hybrid --top-k 3
python -m src.cli query "What is the Transformer architecture and attention mechanism?"
```

## FastAPI REST API

```bash
python main.py
```

| Method | Route                 | Description                                      |
|--------|-----------------------|--------------------------------------------------|
| GET    | `/health`             | Health check                                     |
| GET    | `/ready`              | Readiness probe                                  |
| GET    | `/metrics`            | Index metrics                                    |
| POST   | `/query`              | Full grounded RAG query                          |
| POST   | `/search`             | Raw retrieval with metadata filters              |
| GET    | `/documents`          | List documents                                   |
| GET    | `/documents/{id}`     | Document details                                 |
| DELETE | `/documents/{id}`     | Delete document                                  |
| POST   | `/documents/upload`   | Upload + index                                   |
| POST   | `/documents/ingest`   | Batch ingest                                     |

### Example Query Request (`POST /query`)

```json
{
  "query": "What are the instructions for the NITI Aayog internship application?",
  "metadata_filter": {
    "must": [
      {"field": "category", "operator": "eq", "value": "guidelines"}
    ]
  }
}
```

## Testing & Verification

```bash
# 1. Run the entire unit and integration test suite (88 tests)
pytest tests/ -v

# 2. Run the P0 architectural verification tests
pytest tests/test_production_rag_p0.py -v

# 3. Run the P1 caching, fusion, and metric verification tests
pytest tests/test_production_rag_p1.py -v

# 4. Run the Full Production Benchmark (110 cases)
python -m src.evaluation.benchmark

# 5. Run the Architectural Ablation Study (Configurations A - H)
python -m src.evaluation.ablation --limit 20

# 6. Run the Metadata Filtering Scalability Benchmark
python -m src.evaluation.metadata_benchmark

# 7. Run the Production RAGAS Evaluation Framework
python -m src.evaluation.ragas_eval --engine deterministic
```

## Docker & Deployment

```bash
docker-compose up --build -d
```

Boots:

1. **Standalone Qdrant Vector DB** at port `6333`
2. **Production-oriented Adaptive RAG FastAPI API Server** at port `8000`

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
```
