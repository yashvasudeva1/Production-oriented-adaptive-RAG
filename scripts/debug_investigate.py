import sys
import os
import json
import logging
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("debug_investigate")

from src.components.registry import SystemRegistry
from src.components.ingestion.pipeline import IngestionPipeline
from src.components.chunking.pipeline import ChunkingPipeline
from src.components.indexing.pipeline import IndexingPipeline
from src.components.orchestration.pipeline import RAGOrchestrator
from src.components.keyword_matching import tokenize, normalize_text
from rank_bm25 import BM25Okapi

def run_investigation():
    print("=" * 80)
    print("DEBUG INVESTIGATION: 'How many attention heads does the base model use?'")
    print("=" * 80)

    # 1. SETUP SYSTEM & INGESTION
    pdf_path = Path("documents/1706.03762v7.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found!")
        return

    ingest_pipe = IngestionPipeline()
    chunk_pipe = ChunkingPipeline()

    doc = ingest_pipe.ingest_file(pdf_path)
    chunks = chunk_pipe.chunk_document(doc)
    print(f"\n[STEP 1] Ingestion Check:")
    print(f"PDF parsed: {doc.filename} ({len(doc.blocks)} blocks -> {len(chunks)} chunks)")

    # Save dumped chunks
    dump_path = Path("metadata/debug_chunks_dump.json")
    dump_data = [{
        "id": c.chunk_id,
        "chunk_type": c.chunk_type,
        "page": c.page,
        "text": c.text,
        "section": c.section
    } for c in chunks]
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, indent=2, ensure_ascii=False)
    print(f"Dumped all {len(chunks)} chunks to {dump_path}")

    # Grep targets
    targets = ["h = 8", "h=8", "8 parallel", "parallel attention layers", "d_k", "64"]
    print("\nKeyword grep in chunk texts:")
    target_hits = {}
    for t in targets:
        hits = [c for c in chunks if t.lower() in c.text.lower()]
        target_hits[t] = hits
        hit_ids = [f"{c.chunk_id} (p.{c.page})" for c in hits]
        print(f"  - '{t}': {len(hits)} chunks -> {', '.join(hit_ids) if hits else 'None'}")

    # Identify answer chunks
    ans_chunk_p5 = next((c for c in chunks if "h = 8" in c.text), None)
    ans_chunk_p9 = next((c for c in chunks if "base" in c.text and "2048" in c.text and "64" in c.text), None)
    
    print("\nIdentified Key Evidence Chunks:")
    if ans_chunk_p5:
        print(f"  Evidence Chunk A (Page 5, Section 3.2.2 text): ID={ans_chunk_p5.chunk_id}, page={ans_chunk_p5.page}")
        print(f"    Text: {ans_chunk_p5.text[:200]}...")
    if ans_chunk_p9:
        print(f"  Evidence Chunk B (Page 9, Table 3 Base Row): ID={ans_chunk_p9.chunk_id}, page={ans_chunk_p9.page}")
        print(f"    Text: {ans_chunk_p9.text[:200]}...")

    # Now index into Qdrant & BM25 for this investigation session
    embedder = SystemRegistry.get_embedder()
    qdrant = SystemRegistry.get_qdrant()
    bm25 = SystemRegistry.get_bm25()
    doc_store = SystemRegistry.get_doc_store()

    import numpy as np
    emb_cache_path = Path("metadata/debug_embeddings_1706.03762.npy")
    chunk_texts = [c.text for c in chunks]
    if emb_cache_path.exists():
        print(f"Loading cached embeddings from {emb_cache_path}...")
        embeddings = np.load(emb_cache_path)
    else:
        print("Generating embeddings with SentenceTransformerEmbedder...")
        embeddings = np.array(embedder.embed_texts(chunk_texts))
        np.save(emb_cache_path, embeddings)

    qdrant.upsert_chunks(chunks, embeddings.tolist())
    bm25.upsert_chunks(chunks)
    doc_store.register_document(
        document_id=doc.document_id,
        filename=doc.filename,
        sha256=doc.sha256,
        size_bytes=doc.size_bytes,
        metadata=doc.metadata,
        status="INDEXED"
    )
    print(f"\nIndexed into Qdrant ({qdrant.count()} points) and BM25 ({bm25.count()} chunks)")

    # Orchestrator
    orchestrator = RAGOrchestrator(
        embedder=embedder,
        qdrant_indexer=qdrant,
        bm25_indexer=bm25,
        document_store=doc_store,
    )

    query = "How many attention heads does the base model use?"
    top_k = 10

    # 2. STEP 2: RETRIEVAL CHECK (PER STRATEGY)
    print("\n" + "=" * 80)
    print("[STEP 2] Retrieval Check (Per Strategy) with top_k=10")
    print(f"Query: '{query}'")
    print("=" * 80)

    # A) DENSE ONLY
    dense_resp = orchestrator.dense_retriever.retrieve(query, top_k=top_k)
    print("\n--- 1. Dense-Only Retrieval ---")
    p5_rank_dense = "absent"
    p9_rank_dense = "absent"
    for r in dense_resp.results:
        preview = r.text[:150].replace('\n', ' ')
        print(f"  Rank {r.rank:2d} | Score: {r.score:.4f} | ID: {r.chunk_id} (p.{r.metadata.get('page')}) | {preview}...")
        if ans_chunk_p5 and r.chunk_id == ans_chunk_p5.chunk_id:
            p5_rank_dense = r.rank
        if ans_chunk_p9 and r.chunk_id == ans_chunk_p9.chunk_id:
            p9_rank_dense = r.rank

    print(f"  >> Chunk A (Section 3.2.2, p.5) Rank: {p5_rank_dense}")
    print(f"  >> Chunk B (Table 3 Base, p.9) Rank: {p9_rank_dense}")

    # B) BM25 ONLY
    bm25_hits = bm25.search(query, top_k=top_k)
    print("\n--- 2. BM25-Only Retrieval ---")
    p5_rank_bm25 = "absent"
    p9_rank_bm25 = "absent"
    for r in bm25_hits:
        preview = r.text[:150].replace('\n', ' ')
        print(f"  Rank {r.rank:2d} | Score: {r.score:.4f} | ID: {r.chunk_id} (p.{r.metadata.get('page')}) | {preview}...")
        if ans_chunk_p5 and r.chunk_id == ans_chunk_p5.chunk_id:
            p5_rank_bm25 = r.rank
        if ans_chunk_p9 and r.chunk_id == ans_chunk_p9.chunk_id:
            p9_rank_bm25 = r.rank

    print(f"  >> Chunk A (Section 3.2.2, p.5) Rank: {p5_rank_bm25}")
    print(f"  >> Chunk B (Table 3 Base, p.9) Rank: {p9_rank_bm25}")

    # C) HYBRID (RRF without reranking)
    from src.components.retrieval.fusion import fuse_candidates
    # Run dense and keyword searches as orchestrator does
    dense_cands = orchestrator._search_dense(query, None, top_k)
    kw_cands = orchestrator._search_keyword(query, None, top_k, signals={})
    rrf_fused = fuse_candidates(
        [dense_cands, kw_cands],
        method="rrf",
        k=20,
        top_n=top_k,
    )
    print("\n--- 3. Hybrid (RRF) Retrieval ---")
    p5_rank_hybrid = "absent"
    p9_rank_hybrid = "absent"
    for rank, r in enumerate(rrf_fused, 1):
        preview = r.text[:150].replace('\n', ' ')
        print(f"  Rank {rank:2d} | Score: {r.score:.4f} (dense={r.dense_score}, kw={r.keyword_score}) | ID: {r.chunk_id} (p.{r.metadata.get('page')}) | {preview}...")
        if ans_chunk_p5 and r.chunk_id == ans_chunk_p5.chunk_id:
            p5_rank_hybrid = rank
        if ans_chunk_p9 and r.chunk_id == ans_chunk_p9.chunk_id:
            p9_rank_hybrid = rank

    print(f"  >> Chunk A (Section 3.2.2, p.5) Rank: {p5_rank_hybrid}")
    print(f"  >> Chunk B (Table 3 Base, p.9) Rank: {p9_rank_hybrid}")

    # D) HYBRID_RERANK (RRF + Cross-Encoder)
    reranker = SystemRegistry.get_reranker()
    rerank_results = reranker.rerank(query, rrf_fused, top_k=top_k)
    print("\n--- 4. Hybrid + Cross-Encoder Rerank ---")
    p5_rank_rerank = "absent"
    p9_rank_rerank = "absent"
    for rank, r in enumerate(rerank_results, 1):
        preview = r.text[:150].replace('\n', ' ')
        print(f"  Rank {rank:2d} | Score: {r.score:.4f} (rerank_score={r.rerank_score:.4f}) | ID: {r.chunk_id} (p.{r.metadata.get('page')}) | {preview}...")
        if ans_chunk_p5 and r.chunk_id == ans_chunk_p5.chunk_id:
            p5_rank_rerank = rank
        if ans_chunk_p9 and r.chunk_id == ans_chunk_p9.chunk_id:
            p9_rank_rerank = rank

    print(f"  >> Chunk A (Section 3.2.2, p.5) Rank: {p5_rank_rerank}")
    print(f"  >> Chunk B (Table 3 Base, p.9) Rank: {p9_rank_rerank}")

    # E) BM25 DEEP TOKENIZATION & SCORING BREAKDOWN
    print("\n--- BM25 Tokenization & Per-Term Scoring Breakdown ---")
    q_tokens = tokenize(query)
    print(f"Tokenized query: {q_tokens}")

    # Examine Chunk A (Page 5) and Chunk B (Page 9) in BM25
    bm25_retriever = bm25.retriever
    corpus = bm25_retriever._tokenized_corpus
    chunk_records = bm25_retriever._chunks
    bm25_model = bm25_retriever._bm25  # BM25Okapi instance

    print(f"BM25 Corpus size: {len(corpus)}, avgdl: {bm25_model.avgdl:.2f}")

    for label, target_chunk in [("Chunk A (Section 3.2.2, p.5)", ans_chunk_p5), ("Chunk B (Table 3, p.9)", ans_chunk_p9)]:
        if not target_chunk:
            continue
        print(f"\nAnalyzing {label} [ID: {target_chunk.chunk_id}]:")
        t_tokens = tokenize(target_chunk.text)
        print(f"  Document length in tokens: {len(t_tokens)}")
        print(f"  First 30 tokens: {t_tokens[:30]}")

        # Find doc index in BM25
        doc_idx = next((i for i, c in enumerate(chunk_records) if c.get("chunk_id") == target_chunk.chunk_id), None)
        if doc_idx is not None:
            doc_len = len(corpus[doc_idx])
            total_score = 0.0
            print(f"  Doc index: {doc_idx}, length: {doc_len}")
            print(f"  Per-term score contributions for query tokens:")
            for token in q_tokens:
                if token in bm25_model.idf:
                    idf = bm25_model.idf[token]
                    tf = corpus[doc_idx].count(token)
                    numerator = idf * tf * (bm25_model.k1 + 1)
                    denominator = tf + bm25_model.k1 * (1 - bm25_model.b + bm25_model.b * (doc_len / bm25_model.avgdl))
                    term_score = numerator / denominator if denominator > 0 else 0.0
                    total_score += term_score
                    print(f"    - Token '{token}': tf={tf}, idf={idf:.3f} -> contribution = {term_score:.4f}")
                else:
                    print(f"    - Token '{token}': NOT in BM25 vocabulary (idf=0)")
            print(f"  Total manual BM25 score: {total_score:.4f}")
            # Compare with BM25 model score
            model_scores = bm25_model.get_scores(q_tokens)
            print(f"  Model computed BM25 score: {model_scores[doc_idx]:.4f}")

    # Also analyze Chunk 47 (which talks about varying attention heads)
    ans_chunk_p9_heads = next((c for c in chunks if "In Table 3 rows (A), we vary the number of attention heads" in c.text), None)
    if ans_chunk_p9_heads:
        print(f"\nAnalyzing Distractor Chunk [Chunk 47, ID: {ans_chunk_p9_heads.chunk_id}]:")
        doc_idx = next((i for i, c in enumerate(chunk_records) if c.get("chunk_id") == ans_chunk_p9_heads.chunk_id), None)
        if doc_idx is not None:
            model_scores = bm25_model.get_scores(q_tokens)
            print(f"  Model computed BM25 score: {model_scores[doc_idx]:.4f}")
            for token in q_tokens:
                tf = corpus[doc_idx].count(token)
                idf = bm25_model.idf.get(token, 0.0)
                print(f"    - Token '{token}': tf={tf}, idf={idf:.3f}")

    # 3. STEP 3: ROUTING CHECK
    print("\n" + "=" * 80)
    print("[STEP 3] Routing Check (Adaptive Pipeline)")
    print("=" * 80)

    # Run complete adaptive pipeline query
    res = orchestrator.query(query, mode="adaptive")
    
    print(f"Query Result:")
    print(f"  Answer: {res.answer}")
    print(f"  Abstained: {res.abstained} (Reason: {res.abstention_reason})")
    print(f"  Execution Mode: {res.execution_mode}")
    print(f"  Confidence: {res.confidence:.4f}")
    print(f"  Latency: {res.latency_ms:.2f} ms")
    print(f"\nRetrieval Metadata:")
    print(f"  Cascade stages: {res.retrieval.cascade_stages_executed}")
    print(f"  Dense candidates: {res.retrieval.dense_candidates}")
    print(f"  Keyword candidates: {res.retrieval.keyword_candidates}")
    print(f"  Fused candidates: {res.retrieval.fused_candidates}")
    print(f"  Reranked candidates: {res.retrieval.reranked_candidates}")
    print(f"  Parent expanded count: {res.retrieval.parent_expanded_count}")
    print(f"  Confidence tier: {res.retrieval.confidence_tier}")
    print(f"  Confidence signals (S_i): {res.retrieval.confidence_signals}")
    print(f"  Latency breakdown (ms): {res.retrieval.latency_breakdown_ms}")

    print(f"\nSources returned to LLM ({len(res.sources)}):")
    for idx, s in enumerate(res.sources, 1):
        preview = s.get("snippet", "")[:120].replace('\n', ' ')
        print(f"  Source {idx}: ID={s.get('chunk_id')}, page={s.get('page')}, score={s.get('score')} | {preview}...")

    # Now let's compare with mode='balanced' and mode='hybrid_rerank'
    print("\n" + "=" * 80)
    print("COMPARISON: What happens in mode='balanced'?")
    print("=" * 80)
    res_bal = orchestrator.query(query, mode="balanced")
    print(f"  Answer: {res_bal.answer}")
    print(f"  Abstained: {res_bal.abstained} (Reason: {res_bal.abstention_reason})")
    print(f"  Execution Mode: {res_bal.execution_mode}")
    print(f"  Confidence: {res_bal.confidence:.4f}")
    print(f"  Cascade stages: {res_bal.retrieval.cascade_stages_executed}")
    print(f"  Sources ({len(res_bal.sources)}):")
    for idx, s in enumerate(res_bal.sources, 1):
        preview = s.get("snippet", "")[:120].replace('\n', ' ')
        print(f"    Source {idx}: ID={s.get('chunk_id')}, page={s.get('page')}, score={s.get('score')} | {preview}...")

    print("\n" + "=" * 80)
    print("COMPARISON: What happens in mode='hybrid_rerank'?")
    print("=" * 80)
    res_hr = orchestrator.query(query, mode="hybrid_rerank")
    print(f"  Answer: {res_hr.answer}")
    print(f"  Abstained: {res_hr.abstained} (Reason: {res_hr.abstention_reason})")
    print(f"  Execution Mode: {res_hr.execution_mode}")
    print(f"  Sources ({len(res_hr.sources)}):")
    for idx, s in enumerate(res_hr.sources, 1):
        preview = s.get("snippet", "")[:120].replace('\n', ' ')
        print(f"    Source {idx}: ID={s.get('chunk_id')}, page={s.get('page')}, score={s.get('score')} | {preview}...")

    # Did evidence chunk reach candidate pool before gate?
    # Inspect evidence gate sufficiency details
    ev_gate = orchestrator.evidence_gate
    print(f"\nEvidence Gate Settings:")
    print(f"  min_confidence: {ev_gate.min_confidence}")
    print(f"  min_rerank_threshold: {ev_gate.min_rerank_threshold}")

if __name__ == "__main__":
    run_investigation()
