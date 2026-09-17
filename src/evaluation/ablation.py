from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    citation_accuracy: Optional[float] = None
    abstention_accuracy: Optional[float] = None
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    evaluated_cases: int = 0


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


def evaluate_retrieval_hits(
    retrieved_texts: List[str], expected_keywords: List[str]
) -> Tuple[float, float, float]:
    """Calculate reciprocal rank, recall, and precision for retrieved candidate snippets."""
    if not expected_keywords:
        return 0.0, 0.0, 0.0

    first_rank: Optional[int] = None
    hits_count = 0

    for rank, snippet in enumerate(retrieved_texts, start=1):
        lower_snip = snippet.lower()
        if any(kw.lower() in lower_snip for kw in expected_keywords):
            hits_count += 1
            if first_rank is None:
                first_rank = rank

    rr = 1.0 / first_rank if first_rank is not None else 0.0
    recall = 1.0 if hits_count > 0 else 0.0
    precision = hits_count / max(1, len(retrieved_texts))

    return rr, recall, precision


def run_ablation_suite(
    cases: Optional[List[EvalCase]] = None,
    limit: Optional[int] = None,
    top_k: int = 5,
) -> AblationSuiteResult:
    """
    Run comparative ablation experiment across 6 architectural tiers:
    - Config A: Dense Vector Only
    - Config B: BM25 Keyword Only
    - Config C: Hybrid (Dense + BM25 fused via RRF)
    - Config D: Hybrid + Cross-Encoder Reranker (unconditional)
    - Config E: Adaptive Planner + Hybrid (mode routing without cross-encoder)
    - Config F: Full Adaptive System (Planner + Hybrid + Conditional Reranker + Evidence Gate)
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

    # 1. Config A: Dense Only
    print("Evaluating Config A: Dense Only...")
    latencies_a: List[float] = []
    rrs_a: List[float] = []
    recs_a: List[float] = []
    precs_a: List[float] = []

    for c in answerable_cases:
        t0 = time.perf_counter()
        resp = dense_retriever.retrieve(query=c.query, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        latencies_a.append(lat)

        texts = [hit.text for hit in resp.results]
        rr, rec, prec = evaluate_retrieval_hits(texts, c.expected_doc_keywords)
        rrs_a.append(rr)
        recs_a.append(rec)
        precs_a.append(prec)

    suite_result.configs["A_dense_only"] = ConfigMetrics(
        config_name="A. Dense Only",
        description="Vector similarity search via SentenceTransformers + Qdrant",
        mrr=sum(rrs_a) / max(1, len(rrs_a)),
        recall_at_k=sum(recs_a) / max(1, len(recs_a)),
        precision_at_k=sum(precs_a) / max(1, len(precs_a)),
        mean_latency_ms=sum(latencies_a) / max(1, len(latencies_a)),
        p50_latency_ms=_percentile(latencies_a, 50.0),
        p95_latency_ms=_percentile(latencies_a, 95.0),
        evaluated_cases=len(answerable_cases),
    )

    # 2. Config B: BM25 Only
    print("Evaluating Config B: BM25 Only...")
    latencies_b: List[float] = []
    rrs_b: List[float] = []
    recs_b: List[float] = []
    precs_b: List[float] = []

    for c in answerable_cases:
        t0 = time.perf_counter()
        resp = keyword_retriever.retrieve(query=c.query, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        latencies_b.append(lat)

        texts = [hit.text for hit in resp.results]
        rr, rec, prec = evaluate_retrieval_hits(texts, c.expected_doc_keywords)
        rrs_b.append(rr)
        recs_b.append(rec)
        precs_b.append(prec)

    suite_result.configs["B_bm25_only"] = ConfigMetrics(
        config_name="B. BM25 Only",
        description="Ranked BM25 lexical keyword retrieval with token normalization",
        mrr=sum(rrs_b) / max(1, len(rrs_b)),
        recall_at_k=sum(recs_b) / max(1, len(recs_b)),
        precision_at_k=sum(precs_b) / max(1, len(precs_b)),
        mean_latency_ms=sum(latencies_b) / max(1, len(latencies_b)),
        p50_latency_ms=_percentile(latencies_b, 50.0),
        p95_latency_ms=_percentile(latencies_b, 95.0),
        evaluated_cases=len(answerable_cases),
    )

    # 3. Config C: Hybrid (Dense + BM25 fused via RRF)
    print("Evaluating Config C: Hybrid (Dense + BM25)...")
    latencies_c: List[float] = []
    rrs_c: List[float] = []
    recs_c: List[float] = []
    precs_c: List[float] = []

    for c in answerable_cases:
        t0 = time.perf_counter()
        resp = hybrid_retriever.retrieve(query=c.query, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        latencies_c.append(lat)

        texts = [hit.text for hit in resp.results]
        rr, rec, prec = evaluate_retrieval_hits(texts, c.expected_doc_keywords)
        rrs_c.append(rr)
        recs_c.append(rec)
        precs_c.append(prec)

    suite_result.configs["C_hybrid"] = ConfigMetrics(
        config_name="C. Hybrid (Dense + BM25)",
        description="Parallel dual-stream retrieval fused with Reciprocal Rank Fusion",
        mrr=sum(rrs_c) / max(1, len(rrs_c)),
        recall_at_k=sum(recs_c) / max(1, len(recs_c)),
        precision_at_k=sum(precs_c) / max(1, len(precs_c)),
        mean_latency_ms=sum(latencies_c) / max(1, len(latencies_c)),
        p50_latency_ms=_percentile(latencies_c, 50.0),
        p95_latency_ms=_percentile(latencies_c, 95.0),
        evaluated_cases=len(answerable_cases),
    )

    # 4. Config D: Hybrid + Reranker (Static unconditional)
    print("Evaluating Config D: Hybrid + Reranker (Static)...")
    latencies_d: List[float] = []
    rrs_d: List[float] = []
    recs_d: List[float] = []
    precs_d: List[float] = []

    for c in answerable_cases:
        t0 = time.perf_counter()
        resp = hybrid_retriever.retrieve(query=c.query, top_k=top_k * 2)
        reranked = reranker.rerank(query=c.query, candidates=resp.results, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        latencies_d.append(lat)

        texts = [hit.text for hit in reranked]
        rr, rec, prec = evaluate_retrieval_hits(texts, c.expected_doc_keywords)
        rrs_d.append(rr)
        recs_d.append(rec)
        precs_d.append(prec)

    suite_result.configs["D_hybrid_reranker"] = ConfigMetrics(
        config_name="D. Hybrid + Reranker",
        description="Dual retrieval with unconditional Cross-Encoder re-scoring",
        mrr=sum(rrs_d) / max(1, len(rrs_d)),
        recall_at_k=sum(recs_d) / max(1, len(recs_d)),
        precision_at_k=sum(precs_d) / max(1, len(precs_d)),
        mean_latency_ms=sum(latencies_d) / max(1, len(latencies_d)),
        p50_latency_ms=_percentile(latencies_d, 50.0),
        p95_latency_ms=_percentile(latencies_d, 95.0),
        evaluated_cases=len(answerable_cases),
    )

    # 5. Config E: Adaptive Planner + Hybrid (Bypass Cross-Encoder)
    print("Evaluating Config E: Adaptive Planner + Hybrid...")
    latencies_e: List[float] = []
    rrs_e: List[float] = []
    recs_e: List[float] = []
    precs_e: List[float] = []

    for c in answerable_cases:
        t0 = time.perf_counter()
        plan = full_orchestrator.planner.plan(c.query)
        resp = hybrid_retriever.retrieve(query=plan.normalized_query, top_k=plan.top_k or top_k)
        lat = (time.perf_counter() - t0) * 1000
        latencies_e.append(lat)

        texts = [hit.text for hit in resp.results[:top_k]]
        rr, rec, prec = evaluate_retrieval_hits(texts, c.expected_doc_keywords)
        rrs_e.append(rr)
        recs_e.append(rec)
        precs_e.append(prec)

    suite_result.configs["E_adaptive_planner"] = ConfigMetrics(
        config_name="E. Adaptive Planner + Hybrid",
        description="Dynamic query classification and parameter routing, reranker bypassed",
        mrr=sum(rrs_e) / max(1, len(rrs_e)),
        recall_at_k=sum(recs_e) / max(1, len(recs_e)),
        precision_at_k=sum(precs_e) / max(1, len(precs_e)),
        mean_latency_ms=sum(latencies_e) / max(1, len(latencies_e)),
        p50_latency_ms=_percentile(latencies_e, 50.0),
        p95_latency_ms=_percentile(latencies_e, 95.0),
        evaluated_cases=len(answerable_cases),
    )

    # 6. Config F: Full Adaptive System (End-to-End Orchestrator)
    print("Evaluating Config F: Full Adaptive System (End-to-End)...")
    latencies_f: List[float] = []
    rrs_f: List[float] = []
    recs_f: List[float] = []
    precs_f: List[float] = []
    abstention_hits = 0
    citation_hits = 0
    answerable_count = 0

    for c in test_cases:
        t0 = time.perf_counter()
        rag_resp = full_orchestrator.query(c.query)
        lat = (time.perf_counter() - t0) * 1000
        latencies_f.append(lat)

        if (c.answerable and not rag_resp.abstained) or (not c.answerable and rag_resp.abstained):
            abstention_hits += 1

        if c.answerable:
            answerable_count += 1
            texts = [(s.get("text", "") or s.get("snippet", "")) for s in rag_resp.sources]
            rr, rec, prec = evaluate_retrieval_hits(texts, c.expected_doc_keywords)
            rrs_f.append(rr)
            recs_f.append(rec)
            precs_f.append(prec)

            val = rag_resp.citation_validation
            if val.get("is_valid", False) and len(val.get("invalid_citations", [])) == 0:
                citation_hits += 1

    suite_result.configs["F_full_adaptive"] = ConfigMetrics(
        config_name="F. Full Adaptive System",
        description="End-to-end: Planner + Parallel Hybrid + Conditional Reranker + Evidence Gate",
        mrr=sum(rrs_f) / max(1, len(rrs_f)),
        recall_at_k=sum(recs_f) / max(1, len(recs_f)),
        precision_at_k=sum(precs_f) / max(1, len(precs_f)),
        citation_accuracy=citation_hits / max(1, answerable_count),
        abstention_accuracy=abstention_hits / max(1, len(test_cases)),
        mean_latency_ms=sum(latencies_f) / max(1, len(latencies_f)),
        p50_latency_ms=_percentile(latencies_f, 50.0),
        p95_latency_ms=_percentile(latencies_f, 95.0),
        evaluated_cases=len(test_cases),
    )

    return suite_result


def print_ablation_table(result: AblationSuiteResult) -> None:
    """Print structured Markdown comparison table of ablation configurations."""
    print("\n" + "-" * 95)
    print("ResearchLens — Architectural Ablation Study")
    print("-" * 95)
    header = f"{'Configuration':<28} | {'Recall@5':<8} | {'Prec@5':<7} | {'MRR':<6} | {'Citations':<10} | {'Abstain':<8} | {'Mean (ms)':<9} | {'P95 (ms)':<8}"
    print(header)
    print("-" * 95)

    for key, c in result.configs.items():
        cit_str = f"{c.citation_accuracy * 100:.1f}%" if c.citation_accuracy is not None else "N/A"
        abs_str = f"{c.abstention_accuracy * 100:.1f}%" if c.abstention_accuracy is not None else "N/A"
        row = (
            f"{c.config_name:<28} | "
            f"{c.recall_at_k:<8.4f} | "
            f"{c.precision_at_k:<7.4f} | "
            f"{c.mrr:<6.4f} | "
            f"{cit_str:<10} | "
            f"{abs_str:<8} | "
            f"{c.mean_latency_ms:<9.1f} | "
            f"{c.p95_latency_ms:<8.1f}"
        )
        print(row)
    print("-" * 95)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchLens — Ablation Study Runner")
    parser.add_argument("--limit", type=int, default=20, help="Number of evaluation cases to test")
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
