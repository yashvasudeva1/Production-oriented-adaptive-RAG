# ResearchLens: Production-Grade, Modular, Evidence-Grounded RAG Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ResearchLens** is an enterprise-ready, modular, evidence-grounded Retrieval-Augmented Generation (RAG) platform. Unlike naive RAG demos that simply embed queries and retrieve top-$k$ chunks for an LLM, ResearchLens dynamically analyzes queries, plans retrieval routes, fuses dense and lexical search candidates, reranks with cross-encoders, gates ungrounded premises with an Evidence Gate, generates strictly grounded responses, and validates traceable citations against real documents.

---

## Complete End-to-End Architecture

### 1. User Query & Retrieval Orchestration Pipeline

```
                                USER QUERY
                                    │
                                    ▼
                          Query Normalization
                                    │
                                    ▼
                          Query Categorization
                                    │
                                    ▼
                      Query Metadata Extraction
                                    │
                                    ▼
                             Query Planner
                                    │
                                    ▼
                       Dynamic Retrieval Strategy
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
               Dense             Keyword          Multi-Query
             Retrieval          Retrieval          Retrieval
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                            Metadata Filtering
                                    │
                                    ▼
                           Parent/Child Logic
                                    │
                                    ▼
                         Candidate Fusion (RRF)
                                    │
                                    ▼
                         Cross-Encoder Reranker
                                    │
                                    ▼
                            Context Filtering
                                    │
                                    ▼
                              Evidence Gate ──[FAIL]──► Abstention Response
                                    │
                                 [PASS]
                                    │
                                    ▼
                           Grounded LLM Generator
                                    │
                                    ▼
                            Citation Builder
                                    │
                                    ▼
                           Citation Validator
                                    │
                                    ▼
                              ANSWER & SOURCES
```

### 2. Knowledge Ingestion & Indexing Pipeline

```
                               RAW FILE(S)
                                    │
                                    ▼
                             Ingestion Router
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
              Documents          Images           Audio/Video
          (PDF, DOCX, PPTX)     (OCR PNG)         (Transcripts)
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                             DocumentBlock[]
                                    │
                                    ▼
                         Structure-Aware Chunker
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
               Prose              Code               Tables
          (Hierarchy Path)  (Functions/Classes)  (Repeated Headers)
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                           Parent/Child Chunks
                                    │
                                    ▼
                           Metadata Enrichment
                                    │
                                    ▼
                          Embedding Generation
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                   Qdrant Index            BM25 Index
                  (Dense Vector)       (Lexical Inverted)
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                             Document Store
                         (SHA-256 Deduplication)
```

---

## Key Features & Capabilities

1. **Multi-Format Ingestion**: 16+ formats supported natively (PDF, DOCX, PPTX, XLSX, HTML, Markdown, JSON, CSV, TSV, XML, YAML, EML, Python/code, recursive ZIP archives).
2. **Configurable OCR**: Native image text extraction via Pillow and Tesseract-OCR, supporting custom executable paths (`TESSERACT_CMD`).
3. **Structure-Aware Chunking**: Specialized chunkers for code (function/class aware), markdown (heading hierarchy), tables (repeated schema headers), transcripts (timestamps), and prose (parent-child).
4. **Hybrid Dual Indexing**: Qdrant vector database (embedded in-memory, local file, or standalone server) paired with BM25 inverted lexical indexing.
5. **Adaptive Query Planning**: Classifies queries into a clean taxonomy (`fact`, `detailed`, `general`, `technical_exact`, `comparison`, `metadata_constrained`, `unanswerable`) and routes to the optimal retrieval strategy.
6. **Candidate Fusion**: Reciprocal Rank Fusion (RRF) combines dense and lexical retrieval candidates fairly without score scale skew.
7. **Cross-Encoder Reranking**: Re-evaluates top candidates using sentence-transformers cross-encoders to ensure maximum relevance.
8. **Evidence Gate & Abstention**: Evaluates evidence coverage, relevance, and premise validity. Automatically refuses / abstains from answering false-premise or hallucination-bait queries.
9. **Grounded Generation & Injection Resistance**: Enforces that retrieved documents are treated strictly as passive **DATA**, not instructions. Defeats prompt injections inside ingested documents.
10. **Bidirectional Citation Validation**: Every factual statement is tied to a numbered source `[N]`. The validator guarantees that every citation corresponds to a real, retrieved chunk in the physical document.
11. **Production REST API**: FastAPI server exposing endpoints for file upload, directory ingestion, search, grounded querying, and health monitoring.
12. **CLI Interface**: Full-featured command line tool (`python -m src.cli`) for headless scripting and CI/CD pipelines.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Tesseract-OCR (optional, required for Image OCR):
  - **Windows**: Install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Default: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
  - **Linux**: `sudo apt-get install tesseract-ocr`

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/yashvasudeva1/Production_RAG.git
cd Production_RAG

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

| Variable | Description | Default |
| :--- | :--- | :--- |
| `TESSERACT_CMD` | Path to Tesseract binary | `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| `LLM_PROVIDER` | LLM backend provider (`groq`, `openrouter`, `huggingface`, `offline`) | `groq` |
| `LLM_MODEL` | Open-source model (`qwen/qwen3.8-27b`, `meta-llama/llama-3.1-8b-instruct:free`) | `qwen/qwen3.8-27b` |
| `GROQ_API_KEY` | Groq API Key for ultra-low latency open-source model inference | None |
| `OPENROUTER_API_KEY` | OpenRouter API Key for free open-source models | None |
| `HF_TOKEN` | HuggingFace Inference API Token | None |
| `QDRANT_URL` | Remote Qdrant endpoint (leave empty for embedded) | `""` (embedded `:memory:`) |
| `API_HOST` | FastAPI host | `127.0.0.1` |
| `API_PORT` | FastAPI port | `8000` |

---

## Quickstart & CLI

The CLI provides four core commands: `health`, `ingest`, `search`, and `query`.

### 1. System Health Check
```bash
python -m src.cli health
```
*Output:*
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "indexed_documents": 4,
  "qdrant_points": 36,
  "bm25_chunks": 36
}
```

### 2. Document Ingestion & Indexing
```bash
python -m src.cli ingest ./documents
```
*Output:*
```
Ingesting 4 files from documents...
Parsed 4 documents. Indexing...
Indexed 4 documents (36 chunks) in 3876.6ms. Skipped: 0.
```

### 3. Retrieval Search
```bash
python -m src.cli search "Transformer attention mechanism" --strategy hybrid --top-k 3
```

### 4. Grounded RAG Query
```bash
python -m src.cli query "What is the Transformer architecture and attention mechanism?"
```

*Output:*
```text
------------------------------------------------------------
QUERY TYPE : detailed
CONFIDENCE : 1.00
ABSTAINED  : False
LATENCY    : 1502.0ms
------------------------------------------------------------
ANSWER:
Based on the retrieved evidence:

The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. [1]

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. [2]
------------------------------------------------------------
SOURCES (2):
 [1] NIPS-2017-attention-is-all-you-need-Paper.pdf (locator: page:2)
 [2] NIPS-2017-attention-is-all-you-need-Paper.pdf (locator: page:3)
```

---

## FastAPI REST API

Start the server using `main.py` or uvicorn:
```bash
python main.py
```
*The API is hosted at `http://127.0.0.1:8000` (interactive documentation at `http://127.0.0.1:8000/docs`).*

### API Endpoints

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Cluster, Qdrant, and Document store health |
| `GET` | `/ready` | Readiness probe for Kubernetes / load balancers |
| `GET` | `/metrics` | Document counts and index point metrics |
| `POST` | `/query` | Full grounded RAG query with citations |
| `POST` | `/search` | Raw candidate retrieval (dense, keyword, or hybrid) |
| `GET` | `/documents` | List all ingested and indexed documents |
| `GET` | `/documents/{id}` | Inspect document lifecycle record |
| `DELETE` | `/documents/{id}` | Purge document and chunks from Qdrant and BM25 |
| `POST` | `/documents/upload`| Upload and immediately index a document |
| `POST` | `/documents/ingest`| Batch ingest and index a directory |

### Example Query Request (`POST /query`)
```json
{
  "query": "What are the instructions for the NITI Aayog internship application?",
  "document_ids": null
}
```

### Example Query Response
```json
{
  "query": "What are the instructions for the NITI Aayog internship application?",
  "answer": "Based on the retrieved evidence:\n\nThe application form must be submitted online during the specified monthly periods. [1]",
  "confidence": 1.0,
  "abstained": false,
  "abstention_reason": null,
  "query_type": "detailed",
  "sources": [
    {
      "citation_id": 1,
      "document_id": "doc_1731f7ed91a5404ec0a6",
      "filename": "GENERAL INFORMATION_INTERNSHIP NITI SCHEME.pdf",
      "chunk_id": "chk_3d22a88762cfefa5",
      "page": 1,
      "section": null,
      "source_locator": "page:1",
      "snippet": "Instructions to fill the Online Application Form for NITI Aayog Internship..."
    }
  ],
  "retrieval": {
    "strategy": "detailed",
    "query_type": "detailed",
    "dense_candidates": 12,
    "keyword_candidates": 8,
    "multi_query_candidates": 0,
    "fused_candidates": 16,
    "reranked_candidates": 6,
    "filtered_context_candidates": 4
  },
  "citation_validation": {
    "is_valid": true,
    "cited_ids": [1],
    "valid_citations": [1],
    "invalid_citations": [],
    "hallucinated_citations": [],
    "details": "All citations point to valid retrieved chunks."
  },
  "latency_ms": 312.4
}
```

---

## Testing & Quality Assurance

ResearchLens maintains a comprehensive test suite across all layers.

### 1. Run the Ingestion Suite
```bash
python -X utf8 test_ingestion.py
```
*Expected: 24 PASS, 0 FAIL, 1 SKIP (whisper media test).*

### 2. Run the Full Pytest Suite
```bash
pytest tests/ -v
```
*Expected: 32 PASS, 0 FAIL (100% pass across chunking, indexing, retrieval, query planning, reranker, evidence gate, citations, adversarial, and API).*

### 3. Run the Evaluation Benchmark
```bash
python -m src.evaluation.benchmark --mode fast
```
*Measures MRR, Recall@K, Precision@K, Abstention Accuracy, Citation Accuracy, and latency distribution percentiles (P50, P90, P95, P99).*

### 4. Run the Architectural Ablation Suite
```bash
python -m src.evaluation.ablation --limit 20
```

#### Measured Architectural Ablation Study (20 Cases)

| Configuration | Recall@5 | Prec@5 | MRR | Citation Acc | Abstain Acc | Mean Latency | P95 Latency |
|---|---|---|---|---|---|---|---|
| **A. Dense Only** | 0.8500 | 0.5000 | 0.7767 | N/A | N/A | 12.9 ms | 15.8 ms |
| **B. BM25 Only** | 0.9000 | 0.6200 | 0.9000 | N/A | N/A | 0.3 ms | 0.5 ms |
| **C. Hybrid (Dense + BM25)** | 0.9000 | 0.5700 | 0.8750 | N/A | N/A | 2.8 ms | 3.7 ms |
| **D. Hybrid + Reranker (Static)** | 0.9000 | 0.6000 | 0.8500 | N/A | N/A | 865.7 ms | 956.1 ms |
| **E. Adaptive Planner + Hybrid** | 0.9000 | 0.6100 | 0.8750 | N/A | N/A | 15.3 ms | 18.9 ms |
| **F. Final Additive Cascade** | **0.9000** | **0.6775** | **0.8500** | **95.0%** | **95.0%** | **643.0 ms** | **1240.4 ms** |

---

## Docker & Production Deployment

A complete containerized setup is provided using Docker Compose.

```bash
docker-compose up --build -d
```
This boots:
1. **Standalone Qdrant Vector DB** at port `6333`.
2. **ResearchLens FastAPI API Server** at port `8000`.

---

## Architectural Principles & Design Decisions

1. **Why adaptive retrieval instead of vector-only?**
   Specific queries (e.g. "What was the 2024 revenue in USD?") benefit heavily from exact lexical matching (BM25), whereas conceptual queries ("Explain the self-attention mechanism") require dense embeddings. ResearchLens classifies the intent first and dynamically routes the query.
2. **Why Parent-Child Chunking?**
   Small chunks maximize retrieval relevance score, but starving the LLM of context yields shallow answers. Parent-child chunking indexes child passages for high retrieval precision, then expands to the parent section for rich, coherent answer generation.
3. **Why an Evidence Gate?**
   Conventional RAG answers hallucinate when given false premises or unrelated queries. The Evidence Gate calculates keyword coverage and relevance confidence before generation, abstaining gracefully when evidence is missing.
4. **Why Bidirectional Citation Validation?**
   LLMs frequently fabricate bracketed citations. The citation validator inspects the generated output and verifies that every `[N]` reference resolves to a physical text span in an actual retrieved document block.
5. **Prompt Injection Resistance**:
   Retrieved chunks are treated strictly as untrusted **DATA**. System prompts instruct the LLM that instructions contained within retrieved text must be ignored.
