from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OFFLINE_EVAL", "1")

from ..components.indexing import (
    BM25Indexer,
    DocumentStore,
    QdrantIndexer,
    SentenceTransformerEmbedder,
)
from ..components.orchestration import RAGOrchestrator
from ..components.registry import SystemRegistry
from ..components.reranking import CrossEncoderReranker
from ..components.retrieval import (
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
    RetrievalFilter,
)
from .dataset import EVALUATION_DATASET, EvalCase

logger = logging.getLogger(__name__)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass
class ConfigMetrics:
    config_name: str
    description: str
    mrr: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    citation_accuracy: Optional[float] = None
    abstention_accuracy: Optional[float] = None
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    evaluated_cases: int = 0
    is_pareto_optimal: bool = False


@dataclass
class AblationSuiteResult:
    timestamp: float
    total_dataset_cases: int
    configs: Dict[str, ConfigMetrics] = field(default_factory=dict)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    idx_floor = int(k)
    idx_ceil = min(idx_floor + 1, len(sorted_vals) - 1)
    weight = k - idx_floor
    return sorted_vals[idx_floor] + weight * (sorted_vals[idx_ceil] - sorted_vals[idx_floor])


def compute_ndcg_at_k(relevance_scores: List[float], k: int = 5) -> float:
    if not relevance_scores:
        return 0.0
    k_scores = relevance_scores[:k]
    dcg = sum((rel / math.log2(idx + 1)) for idx, rel in enumerate(k_scores, start=1))
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((rel / math.log2(idx + 1)) for idx, rel in enumerate(ideal_scores, start=1))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval_hits(
    retrieved_texts: List[str],
    case: EvalCase,
    sources: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
) -> Tuple[float, float, float, float]:
    """Calculate reciprocal rank, recall, precision, and nDCG for retrieved candidates."""
    if not case.expected_doc_keywords and not case.expected_document_ids and not case.expected_chunk_ids:
        return 0.0, 0.0, 0.0, 0.0

    first_rank: Optional[int] = None
    hits_count = 0
    rel_scores: List[float] = []

    for rank, snippet in enumerate(retrieved_texts, start=1):
        lower_snip = snippet.lower()
        src = (sources[rank - 1] if sources and rank - 1 < len(sources) else {})
        kw_match = any(kw.lower() in lower_snip for kw in case.expected_doc_keywords)
        doc_match = bool(
            getattr(case, "expected_document_ids", None)
            and any(str(did).lower() == str(src.get("document_id", "")).lower() for did in case.expected_document_ids)
        )
        chk_match = bool(
            getattr(case, "expected_chunk_ids", None)
            and any(str(cid).lower() == str(src.get("chunk_id", "")).lower() for cid in case.expected_chunk_ids)
        )

        is_hit = kw_match or doc_match or chk_match
        rel_scores.append(1.0 if is_hit else 0.0)

        if is_hit:
            hits_count += 1
            if first_rank is None:
                first_rank = rank

    rr = 1.0 / first_rank if first_rank is not None else 0.0
    recall = 1.0 if hits_count > 0 else 0.0
    precision = hits_count / max(1, len(retrieved_texts))
    ndcg = compute_ndcg_at_k(rel_scores, k=k)

    return rr, recall, precision, ndcg


def calculate_pareto_frontier(configs: Dict[str, ConfigMetrics]) -> None:
    """
    Determine Pareto optimality: a configuration is Pareto-optimal if no other configuration
    achieves strictly higher quality (Recall@K / MRR) with lower or equal latency (P50).
    """
    for key, c in configs.items():
        dominated = False
        for other_key, other in configs.items():
            if other_key == key:
                continue
            better_quality = (other.recall_at_k >= c.recall_at_k) and (other.mrr >= c.mrr)
            better_latency = (other.p50_latency_ms <= c.p50_latency_ms) and (other.mean_latency_ms <= c.mean_latency_ms)
            # FIX: previously omitted mean_latency_ms from strictly_better
            strictly_better = (
                other.recall_at_k > c.recall_at_k
                or other.mrr > c.mrr
                or other.p50_latency_ms < c.p50_latency_ms
                or other.mean_latency_ms < c.mean_latency_ms
            )
            if better_quality and better_latency and strictly_better:
                dominated = True
                break
        c.is_pareto_optimal = not dominated


def run_ablation_suite(
    cases: Optional[List[EvalCase]] = None,
    limit: Optional[int] = None,
    top_k: int = 5,
) -> AblationSuiteResult:
    """
    Run comprehensive architectural ablation study across 8 configurations:
    - Config A: BM25 Keyword Only (Ultra-cheap lexical baseline)
    - Config B: Dense Vector Only (Dense semantic baseline)
    - Config C: Hybrid (Dense + BM25 fused via RRF, no filters, no reranking)
    - Config D: Hybrid + Hard Metadata Filter (Filter-aware hybrid)
    - Config E: Hybrid + Hard Metadata Filter + Unconditional Cross-Encoder Reranker
    - Config F: Adaptive Cost-Aware Routing (Fast/Balanced/Deep without cross-encoder)
    - Config G: Full System with Conditional Reranking & Controlled Late Parent Expansion
    - Config H: Full System + Multi-Level Caching Warm Run (Cached production deployment)
    """
    test_cases = cases or EVALUATION_DATASET
    if limit is not None and limit > 0:
        test_cases = test_cases[:limit]

    answerable_cases = [c for c in test_cases if c.answerable]

    # Shared singletons from SystemRegistry
    embedder = SystemRegistry.get_embedder()
    qdrant = SystemRegistry.get_qdrant()
    bm25 = SystemRegistry.get_bm25()
    doc_store = SystemRegistry.get_doc_store()
    reranker = SystemRegistry.get_reranker()

    # Backfill vector index if necessary
    if qdrant.count() == 0 and bm25.count() > 0:
        from ..components.chunking.models import Chunk
        loaded_chunks = [Chunk.from_dict(c) for c in bm25._all_chunks]
        vecs = embedder.embed_texts([c.text for c in loaded_chunks])
        qdrant.upsert_chunks(loaded_chunks, vecs)

    dense_retriever = DenseRetriever(indexer=qdrant, embedder=embedder)
    keyword_retriever = KeywordRetriever(indexer=bm25)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever, keyword_retriever=keyword_retriever
    )
    full_orchestrator = RAGOrchestrator(
        embedder=embedder,
        qdrant_indexer=qdrant,
        bm25_indexer=bm25,
        document_store=doc_store,
        reranker=reranker,
    )

    suite_result = AblationSuiteResult(
        timestamp=time.time(),
        total_dataset_cases=len(test_cases),
    )

    # -------------------------------------------------------------
    # 1. Config A: BM25 Only
    # -------------------------------------------------------------
    print("Evaluating Config A: BM25 Only...")
    lats_a, rrs_a, recs_a, precs_a, ndcgs_a = [], [], [], [], []
    for c in answerable_cases:
        t0 = time.perf_counter()
        resp = keyword_retriever.retrieve(query=c.query, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        lats_a.append(lat)
        texts = [hit.text for hit in resp.results]
        rr, rec, prec, ndcg = evaluate_retrieval_hits(texts, c, k=top_k)
        rrs_a.append(rr)
        recs_a.append(rec)
        precs_a.append(prec)
        ndcgs_a.append(ndcg)

    suite_result.configs["A_bm25_only"] = ConfigMetrics(
        config_name="A. BM25 Only",
        description="Ranked BM25 lexical keyword retrieval baseline",
        mrr=sum(rrs_a) / max(1, len(rrs_a)),
        recall_at_k=sum(recs_a) / max(1, len(recs_a)),
        precision_at_k=sum(precs_a) / max(1, len(precs_a)),
        ndcg_at_k=sum(ndcgs_a) / max(1, len(ndcgs_a)),
        mean_latency_ms=sum(lats_a) / max(1, len(lats_a)),
        p50_latency_ms=_percentile(lats_a, 50.0),
        p95_latency_ms=_percentile(lats_a, 95.0),
        p99_latency_ms=_percentile(lats_a, 99.0),
        evaluated_cases=len(answerable_cases),
    )

    # -------------------------------------------------------------
    # 2. Config B: Dense Vector Only
    # -------------------------------------------------------------
    print("Evaluating Config B: Dense Vector Only...")
    lats_b, rrs_b, recs_b, precs_b, ndcgs_b = [], [], [], [], []
    for c in answerable_cases:
        t0 = time.perf_counter()
        resp = dense_retriever.retrieve(query=c.query, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        lats_b.append(lat)
        texts = [hit.text for hit in resp.results]
        rr, rec, prec, ndcg = evaluate_retrieval_hits(texts, c, k=top_k)
        rrs_b.append(rr)
        recs_b.append(rec)
        precs_b.append(prec)
        ndcgs_b.append(ndcg)

    suite_result.configs["B_dense_only"] = ConfigMetrics(
        config_name="B. Dense Only",
        description="Vector similarity search via SentenceTransformers + Qdrant",
        mrr=sum(rrs_b) / max(1, len(rrs_b)),
        recall_at_k=sum(recs_b) / max(1, len(recs_b)),
        precision_at_k=sum(precs_b) / max(1, len(precs_b)),
        ndcg_at_k=sum(ndcgs_b) / max(1, len(ndcgs_b)),
        mean_latency_ms=sum(lats_b) / max(1, len(lats_b)),
        p50_latency_ms=_percentile(lats_b, 50.0),
        p95_latency_ms=_percentile(lats_b, 95.0),
        p99_latency_ms=_percentile(lats_b, 99.0),
        evaluated_cases=len(answerable_cases),
    )

    # -------------------------------------------------------------
    # 3. Config C: Hybrid (Dense + BM25 via RRF)
    # -------------------------------------------------------------
    print("Evaluating Config C: Hybrid (Dense + BM25 via RRF)...")
    lats_c, rrs_c, recs_c, precs_c, ndcgs_c = [], [], [], [], []
    for c in answerable_cases:
        t0 = time.perf_counter()
        resp = hybrid_retriever.retrieve(query=c.query, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        lats_c.append(lat)
        texts = [hit.text for hit in resp.results]
        rr, rec, prec, ndcg = evaluate_retrieval_hits(texts, c, k=top_k)
        rrs_c.append(rr)
        recs_c.append(rec)
        precs_c.append(prec)
        ndcgs_c.append(ndcg)

    suite_result.configs["C_hybrid"] = ConfigMetrics(
        config_name="C. Hybrid (Dense + BM25)",
        description="Parallel dual-stream retrieval fused with Reciprocal Rank Fusion",
        mrr=sum(rrs_c) / max(1, len(rrs_c)),
        recall_at_k=sum(recs_c) / max(1, len(recs_c)),
        precision_at_k=sum(precs_c) / max(1, len(precs_c)),
        ndcg_at_k=sum(ndcgs_c) / max(1, len(ndcgs_c)),
        mean_latency_ms=sum(lats_c) / max(1, len(lats_c)),
        p50_latency_ms=_percentile(lats_c, 50.0),
        p95_latency_ms=_percentile(lats_c, 95.0),
        p99_latency_ms=_percentile(lats_c, 99.0),
        evaluated_cases=len(answerable_cases),
    )

    # -------------------------------------------------------------
    # 4. Config D: Hybrid + Hard Metadata Filter
    # -------------------------------------------------------------
    print("Evaluating Config D: Hybrid + Hard Metadata Filter...")
    lats_d, rrs_d, recs_d, precs_d, ndcgs_d = [], [], [], [], []
    for c in answerable_cases:
        t0 = time.perf_counter()
        plan = full_orchestrator.planner.plan(c.query)
        m_filter = plan.retrieval_filter
        resp = hybrid_retriever.retrieve(query=plan.normalized_query, metadata_filter=m_filter, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        lats_d.append(lat)
        texts = [hit.text for hit in resp.results]
        rr, rec, prec, ndcg = evaluate_retrieval_hits(texts, c, k=top_k)
        rrs_d.append(rr)
        recs_d.append(rec)
        precs_d.append(prec)
        ndcgs_d.append(ndcg)

    suite_result.configs["D_hybrid_filter"] = ConfigMetrics(
        config_name="D. Hybrid + Filter",
        description="Dual retrieval with native Qdrant & BM25 pre-filtering",
        mrr=sum(rrs_d) / max(1, len(rrs_d)),
        recall_at_k=sum(recs_d) / max(1, len(recs_d)),
        precision_at_k=sum(precs_d) / max(1, len(precs_d)),
        ndcg_at_k=sum(ndcgs_d) / max(1, len(ndcgs_d)),
        mean_latency_ms=sum(lats_d) / max(1, len(lats_d)),
        p50_latency_ms=_percentile(lats_d, 50.0),
        p95_latency_ms=_percentile(lats_d, 95.0),
        p99_latency_ms=_percentile(lats_d, 99.0),
        evaluated_cases=len(answerable_cases),
    )

    # -------------------------------------------------------------
    # 5. Config E: Hybrid + Filter + Unconditional Cross-Encoder Reranker
    # -------------------------------------------------------------
    print("Evaluating Config E: Hybrid + Filter + Unconditional Reranker...")
    lats_e, rrs_e, recs_e, precs_e, ndcgs_e = [], [], [], [], []
    for c in answerable_cases:
        t0 = time.perf_counter()
        plan = full_orchestrator.planner.plan(c.query)
        m_filter = plan.retrieval_filter
        resp = hybrid_retriever.retrieve(query=plan.normalized_query, metadata_filter=m_filter, top_k=top_k * 2)
        reranked = reranker.rerank(query=plan.normalized_query, candidates=resp.results, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        lats_e.append(lat)
        texts = [hit.text for hit in reranked]
        rr, rec, prec, ndcg = evaluate_retrieval_hits(texts, c, k=top_k)
        rrs_e.append(rr)
        recs_e.append(rec)
        precs_e.append(prec)
        ndcgs_e.append(ndcg)

    suite_result.configs["E_unconditional_rerank"] = ConfigMetrics(
        config_name="E. Hybrid + Filter + Static Rerank",
        description="Dual retrieval with mandatory Cross-Encoder on all queries",
        mrr=sum(rrs_e) / max(1, len(rrs_e)),
        recall_at_k=sum(recs_e) / max(1, len(recs_e)),
        precision_at_k=sum(precs_e) / max(1, len(precs_e)),
        ndcg_at_k=sum(ndcgs_e) / max(1, len(ndcgs_e)),
        mean_latency_ms=sum(lats_e) / max(1, len(lats_e)),
        p50_latency_ms=_percentile(lats_e, 50.0),
        p95_latency_ms=_percentile(lats_e, 95.0),
        p99_latency_ms=_percentile(lats_e, 99.0),
        evaluated_cases=len(answerable_cases),
    )

    # -------------------------------------------------------------
    # 6. Config F: Adaptive Cost-Aware Routing (No Cross-Encoder)
    # -------------------------------------------------------------
    print("Evaluating Config F: Adaptive Routing (No Cross-Encoder)...")
    lats_f, rrs_f, recs_f, precs_f, ndcgs_f = [], [], [], [], []
    for c in answerable_cases:
        t0 = time.perf_counter()
        plan = full_orchestrator.planner.plan(c.query)
        if plan.execution_mode == "fast":
            resp = keyword_retriever.retrieve(query=plan.normalized_query, metadata_filter=plan.retrieval_filter, top_k=top_k)
            texts = [hit.text for hit in resp.results]
        else:
            resp = hybrid_retriever.retrieve(query=plan.normalized_query, metadata_filter=plan.retrieval_filter, top_k=top_k)
            texts = [hit.text for hit in resp.results]
        lat = (time.perf_counter() - t0) * 1000
        lats_f.append(lat)
        rr, rec, prec, ndcg = evaluate_retrieval_hits(texts, c, k=top_k)
        rrs_f.append(rr)
        recs_f.append(rec)
        precs_f.append(prec)
        ndcgs_f.append(ndcg)

    suite_result.configs["F_adaptive_routing"] = ConfigMetrics(
        config_name="F. Adaptive Routing (No Rerank)",
        description="Cost-aware mode routing (Fast/Balanced/Deep) without cross-encoder",
        mrr=sum(rrs_f) / max(1, len(rrs_f)),
        recall_at_k=sum(recs_f) / max(1, len(recs_f)),
        precision_at_k=sum(precs_f) / max(1, len(precs_f)),
        ndcg_at_k=sum(ndcgs_f) / max(1, len(ndcgs_f)),
        mean_latency_ms=sum(lats_f) / max(1, len(lats_f)),
        p50_latency_ms=_percentile(lats_f, 50.0),
        p95_latency_ms=_percentile(lats_f, 95.0),
        p99_latency_ms=_percentile(lats_f, 99.0),
        evaluated_cases=len(answerable_cases),
    )

    # -------------------------------------------------------------
    # 7. Config G: Full System (Conditional Reranking + Parent Expansion + Gate)
    # -------------------------------------------------------------
    print("Evaluating Config G: Full System with Conditional Reranking...")
    lats_g, rrs_g, recs_g, precs_g, ndcgs_g = [], [], [], [], []
    abstention_hits_g = 0
    citation_hits_g = 0
    answerable_count_g = 0

    for c in test_cases:
        t0 = time.perf_counter()
        rag_resp = full_orchestrator.query(c.query)
        lat = (time.perf_counter() - t0) * 1000
        lats_g.append(lat)

        if (c.answerable and not rag_resp.abstained) or (not c.answerable and rag_resp.abstained):
            abstention_hits_g += 1

        if c.answerable:
            answerable_count_g += 1
            texts = [(s.get("text", "") or s.get("snippet", "")) for s in rag_resp.sources]
            rr, rec, prec, ndcg = evaluate_retrieval_hits(texts, c, sources=rag_resp.sources, k=top_k)
            rrs_g.append(rr)
            recs_g.append(rec)
            precs_g.append(prec)
            ndcgs_g.append(ndcg)

            val = rag_resp.citation_validation
            if val.get("is_valid", False) and len(val.get("invalid_citations", [])) == 0:
                citation_hits_g += 1

    suite_result.configs["G_full_conditional"] = ConfigMetrics(
        config_name="G. Full System (Conditional Rerank)",
        description="End-to-End: Adaptive Routing + Conditional Reranker + Evidence Gate",
        mrr=sum(rrs_g) / max(1, len(rrs_g)),
        recall_at_k=sum(recs_g) / max(1, len(recs_g)),
        precision_at_k=sum(precs_g) / max(1, len(precs_g)),
        ndcg_at_k=sum(ndcgs_g) / max(1, len(ndcgs_g)),
        citation_accuracy=citation_hits_g / max(1, answerable_count_g),
        abstention_accuracy=abstention_hits_g / max(1, len(test_cases)),
        mean_latency_ms=sum(lats_g) / max(1, len(lats_g)),
        p50_latency_ms=_percentile(lats_g, 50.0),
        p95_latency_ms=_percentile(lats_g, 95.0),
        p99_latency_ms=_percentile(lats_g, 99.0),
        evaluated_cases=len(test_cases),
    )

    # -------------------------------------------------------------
    # 8. Config H: Full System + Multi-Level Caching (Warm Deployment)
    # -------------------------------------------------------------
    print("Evaluating Config H: Full System + Caching (Warm Deployment)...")
    lats_h, rrs_h, recs_h, precs_h, ndcgs_h = [], [], [], [], []
    # Config H tracks its own citation/abstention metrics (not reusing G's counters)
    abstention_hits_h = 0
    citation_hits_h = 0
    answerable_count_h = 0
    for c in test_cases:
        t0 = time.perf_counter()
        rag_resp = full_orchestrator.query(c.query)
        lat = (time.perf_counter() - t0) * 1000
        lats_h.append(lat)

        if (c.answerable and not rag_resp.abstained) or (not c.answerable and rag_resp.abstained):
            abstention_hits_h += 1

        if c.answerable:
            answerable_count_h += 1
            texts = [(s.get("text", "") or s.get("snippet", "")) for s in rag_resp.sources]
            rr, rec, prec, ndcg = evaluate_retrieval_hits(texts, c, sources=rag_resp.sources, k=top_k)
            rrs_h.append(rr)
            recs_h.append(rec)
            precs_h.append(prec)
            ndcgs_h.append(ndcg)
            val = rag_resp.citation_validation
            if val.get("is_valid", False) and len(val.get("invalid_citations", [])) == 0:
                citation_hits_h += 1

    suite_result.configs["H_full_cached_warm"] = ConfigMetrics(
        config_name="H. Full System + Cache [Warm]",
        description="Full adaptive system with multi-level embedding and retrieval cache hit",
        mrr=sum(rrs_h) / max(1, len(rrs_h)),
        recall_at_k=sum(recs_h) / max(1, len(recs_h)),
        precision_at_k=sum(precs_h) / max(1, len(precs_h)),
        ndcg_at_k=sum(ndcgs_h) / max(1, len(ndcgs_h)),
        citation_accuracy=citation_hits_h / max(1, answerable_count_h),
        abstention_accuracy=abstention_hits_h / max(1, len(test_cases)),
        mean_latency_ms=sum(lats_h) / max(1, len(lats_h)),
        p50_latency_ms=_percentile(lats_h, 50.0),
        p95_latency_ms=_percentile(lats_h, 95.0),
        p99_latency_ms=_percentile(lats_h, 99.0),
        evaluated_cases=len(test_cases),
    )

    # Compute Pareto frontier across all configurations
    calculate_pareto_frontier(suite_result.configs)

    return suite_result


def print_ablation_table(result: AblationSuiteResult) -> None:
    """Print structured comparison table with Pareto frontier analysis."""
    print("\n" + "=" * 115)
    print("Production-oriented Adaptive RAG — Architectural Ablation Study & Pareto Frontier Analysis")
    print("=" * 115)
    header = (
        f"{'Configuration':<34} | {'Recall@5':<8} | {'Prec@5':<7} | {'MRR':<6} | "
        f"{'nDCG@5':<7} | {'P50 (ms)':<8} | {'P95 (ms)':<8} | {'Mean (ms)':<9} | {'Pareto'}"
    )
    print(header)
    print("-" * 115)

    for key, c in result.configs.items():
        pareto_mark = "[PARETO]" if c.is_pareto_optimal else ""
        row = (
            f"{c.config_name:<34} | "
            f"{c.recall_at_k:<8.4f} | "
            f"{c.precision_at_k:<7.4f} | "
            f"{c.mrr:<6.4f} | "
            f"{c.ndcg_at_k:<7.4f} | "
            f"{c.p50_latency_ms:<8.1f} | "
            f"{c.p95_latency_ms:<8.1f} | "
            f"{c.mean_latency_ms:<9.1f} | "
            f"{pareto_mark}"
        )
        print(row)
    print("=" * 115)
    print("Note: [PARETO] marks Pareto-optimal designs on the Quality (Recall/MRR) vs Latency frontier.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-oriented Adaptive RAG — Ablation Study Runner")
    parser.add_argument("--limit", type=int, default=10, help="Number of evaluation cases to test")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieval depth")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for ablation results")

    args = parser.parse_args()

    print(f"Starting ablation suite with limit={args.limit}, top_k={args.top_k}...")
    res = run_ablation_suite(limit=args.limit, top_k=args.top_k)
    print_ablation_table(res)

    if args.output:
        out_dict = asdict(res)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Ablation results saved to: {args.output}")


if __name__ == "__main__":
    main()
