import sys
import os
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Force offline deterministic evaluation for reproducibility and speed
os.environ["OFFLINE_EVAL"] = "1"
sys.stdout.reconfigure(encoding='utf-8')

from src.components.registry import SystemRegistry
from src.components.chunking.models import Chunk
from src.components.orchestration.pipeline import RAGOrchestrator
from src.evaluation.dataset import EVALUATION_DATASET

def compute_metrics(case, retrieved_chunks):
    # Metric calculation matching run_standardized_benchmark.py
    keywords = [kw.lower() for kw in case.expected_doc_keywords]
    top_5 = retrieved_chunks[:5]
    
    # Hit@5
    hit = 0.0
    for c in top_5:
        text = c.text.lower() if hasattr(c, 'text') else str(c).lower()
        if any(kw in text for kw in keywords):
            hit = 1.0
            break
            
    # Recall@5
    all_matched = set()
    for c in top_5:
        text = c.text.lower() if hasattr(c, 'text') else str(c).lower()
        for kw in keywords:
            if kw in text:
                all_matched.add(kw)
    recall = len(all_matched) / max(1, len(keywords))
    
    return hit, recall

def run_comparison():
    print("=" * 80)
    print("RUNNING 110-CASE BENCHMARK COMPARISON: ADAPTIVE vs BALANCED")
    print("=" * 80)

    orchestrator = RAGOrchestrator()
    # Sync Qdrant if needed
    if orchestrator.qdrant.count() == 0 and orchestrator.bm25.count() > 0:
        print("Synchronizing Qdrant collection with BM25 corpus...")
        loaded = [Chunk.from_dict(c) for c in orchestrator.bm25._all_chunks]
        vecs = orchestrator.embedder.embed_texts([c.text for c in loaded])
        orchestrator.qdrant.upsert_chunks(loaded, vecs)
        
    print(f"Corpus size: {orchestrator.qdrant.count()} vectors, {orchestrator.bm25.count()} BM25 chunks")
    print(f"Total test cases: {len(EVALUATION_DATASET)}")

    results = []
    
    for idx, case in enumerate(EVALUATION_DATASET, 1):
        # 1. Run in ADAPTIVE mode
        res_adapt = orchestrator.query(case.query, mode="adaptive")
        adapt_route = res_adapt.execution_mode
        adapt_stages = res_adapt.retrieval.cascade_stages_executed
        # Get top-5 chunks
        adapt_chunks = res_adapt.sources or []
        # Fallback to retrieved chunks if sources is formatted
        adapt_hit, adapt_recall = compute_metrics(case, adapt_chunks)
        
        # 2. Run in BALANCED mode
        res_bal = orchestrator.query(case.query, mode="balanced")
        bal_chunks = res_bal.sources or []
        bal_hit, bal_recall = compute_metrics(case, bal_chunks)
        
        results.append({
            "idx": idx,
            "id": case.query_id,
            "category": case.category,
            "query": case.query,
            "expected_keywords": case.expected_doc_keywords,
            "adapt_route": adapt_route,
            "adapt_stages": adapt_stages,
            "adapt_hit": adapt_hit,
            "adapt_recall": adapt_recall,
            "bal_hit": bal_hit,
            "bal_recall": bal_recall,
            "adapt_confidence": res_adapt.confidence,
            "bal_confidence": res_bal.confidence,
        })

    # Filter to cases where route=FAST and (recall@5 or hit@5) is lower than balanced
    fast_cases = [r for r in results if r["adapt_route"] == "fast" or "fast_bm25" in r["adapt_stages"]]
    regressions = [
        r for r in fast_cases
        if r["adapt_recall"] < r["bal_recall"] or r["adapt_hit"] < r["bal_hit"]
    ]

    print("\n" + "=" * 80)
    print(f"SUMMARY OF FINDINGS:")
    print(f"Total evaluated queries: {len(results)}")
    print(f"Queries routed to FAST (or executed fast_bm25): {len(fast_cases)}")
    print(f"Queries where FAST has lower Recall@5 or Hit@5 than BALANCED: {len(regressions)}")
    print("=" * 80)

    if regressions:
        print("\nDETAILED LIST OF REGRESSED QUERIES (FAST < BALANCED):")
        for i, reg in enumerate(regressions, 1):
            print(f"\n[{i}] Query ID: {reg['id']} | Category: {reg['category']}")
            print(f"    Query: \"{reg['query']}\"")
            print(f"    Keywords: {reg['expected_keywords']}")
            print(f"    ADAPTIVE (FAST): Hit@5={reg['adapt_hit']:.2f}, Recall@5={reg['adapt_recall']:.2f}, Conf={reg['adapt_confidence']:.2f}")
            print(f"    BALANCED:       Hit@5={reg['bal_hit']:.2f}, Recall@5={reg['bal_recall']:.2f}, Conf={reg['bal_confidence']:.2f}")
    else:
        print("\nNo regressions found in the 110 canonical test cases where FAST was selected.")

    # Also list all fast cases so we can inspect their patterns
    print("\n" + "=" * 80)
    print("ALL FAST CASES ROUTED IN ADAPTIVE MODE:")
    print("=" * 80)
    for r in fast_cases:
        status = "REGRESSED" if (r["adapt_recall"] < r["bal_recall"] or r["adapt_hit"] < r["bal_hit"]) else "MATCHED/EQUAL"
        print(f"[{r['id']}] ({r['category']}) '{r['query'][:55]}...' -> {status} (Adapt Rec={r['adapt_recall']:.2f}, Bal Rec={r['bal_recall']:.2f})")

if __name__ == "__main__":
    run_comparison()
