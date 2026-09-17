from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from src.components.registry import SystemRegistry
from src.components.retrieval import DenseRetriever, KeywordRetriever, HybridRetriever
from src.components.orchestration import RAGOrchestrator
from src.evaluation.dataset import EVALUATION_DATASET

# Use first 20 evaluation cases (exact subset used in ablation)
cases = EVALUATION_DATASET[:20]

embedder = SystemRegistry.get_embedder()
qdrant = SystemRegistry.get_qdrant()
bm25 = SystemRegistry.get_bm25()
doc_store = SystemRegistry.get_doc_store()
reranker = SystemRegistry.get_reranker()

orchestrator = RAGOrchestrator(
    embedder=embedder,
    qdrant_indexer=qdrant,
    bm25_indexer=bm25,
    document_store=doc_store,
    reranker=reranker,
)

print("-" * 80)
print(f"TRACING {len(cases)} EVALUATION CASES")
print("-" * 80)

def find_rank(texts: List[str], keywords: List[str]) -> Optional[int]:
    for idx, text in enumerate(texts, start=1):
        lower = text.lower()
        if any(kw.lower() in lower for kw in keywords):
            return idx
    return None

loss_cases = []

for idx, case in enumerate(cases, start=1):
    if not case.answerable:
        continue

    query = case.query
    kws = case.expected_doc_keywords

    # 1. Fast BM25 candidates
    bm25_hits = orchestrator._search_keyword(query, None, top_k=5, signals={})
    bm25_texts = [h.text for h in bm25_hits]
    rank_fast = find_rank(bm25_texts, kws)

    # 2. Balanced Hybrid candidates
    hybrid_hits = orchestrator.hybrid_retriever.retrieve(query=query, top_k=5).results
    hybrid_texts = [h.text for h in hybrid_hits[:5]]
    rank_balanced = find_rank(hybrid_texts, kws)

    # 3. Full Adaptive pipeline response
    resp = orchestrator.query(query)
    final_sources = [s.get("text", s.get("snippet", "")) for s in resp.sources]
    rank_final = find_rank(final_sources, kws)

    is_lost = (rank_fast is not None and rank_fast <= 5 and (rank_final is None or rank_final > 5))

    status = "RETAINED" if not is_lost else "LOST!"
    print(f"[{case.query_id:10s}] FastRank: {str(rank_fast):>4s} | BalRank: {str(rank_balanced):>4s} | FinalRank: {str(rank_final):>4s} | Mode: {resp.execution_mode:8s} | Stages: {resp.retrieval.cascade_stages_executed} | {status}")
    if is_lost or (rank_fast is not None and rank_final is not None and rank_final > rank_fast):
        loss_cases.append({
            "query_id": case.query_id,
            "query": case.query,
            "keywords": kws,
            "rank_fast": rank_fast,
            "rank_balanced": rank_balanced,
            "rank_final": rank_final,
            "mode": resp.execution_mode,
            "stages": resp.retrieval.cascade_stages_executed,
            "escalation_reason": resp.retrieval.escalation_reason,
            "sources_snippets": [s[:100] for s in final_sources],
            "bm25_top_snippet": bm25_texts[0][:100] if bm25_texts else "",
        })

print("-" * 80)
print(f"TOTAL LOST OR DEGRADED CASES: {len(loss_cases)}")
print("-" * 80)
for lc in loss_cases:
    print(f"\nQUERY ID: {lc['query_id']}")
    print(f"QUERY   : {lc['query']}")
    print(f"KWS     : {lc['keywords']}")
    print(f"FastRank: {lc['rank_fast']} -> BalRank: {lc['rank_balanced']} -> FinalRank: {lc['rank_final']}")
    print(f"Mode    : {lc['mode']}, Stages: {lc['stages']}")
    print(f"Escalate: {lc['escalation_reason']}")
    print(f"BM25 Top: {lc['bm25_top_snippet']}")
    print(f"Final 1 : {lc['sources_snippets'][0] if lc['sources_snippets'] else 'NONE'}")
