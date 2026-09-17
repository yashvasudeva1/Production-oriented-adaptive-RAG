from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..components.orchestration import RAGOrchestrator
from ..components.registry import SystemRegistry
from .dataset import EVALUATION_DATASET, EvalCase

logger = logging.getLogger(__name__)

# Ensure utf-8 output for terminal printing
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BENCHMARK_CASES: List[EvalCase] = EVALUATION_DATASET


@dataclass
class BenchmarkReport:
    total_cases: int = 0
    mrr: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    abstention_accuracy: float = 0.0
    citation_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Routing metrics
    fast_path_rate: float = 0.0
    balanced_path_rate: float = 0.0
    deep_path_rate: float = 0.0
    escalation_rate: float = 0.0
    rerank_rate: float = 0.0
    multi_query_rate: float = 0.0
    parent_expansion_rate: float = 0.0
    abstention_rate: float = 0.0

    category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)


def _compute_percentile(values: List[float], p: float) -> float:
    """Calculate percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    idx_floor = int(k)
    idx_ceil = min(idx_floor + 1, len(sorted_vals) - 1)
    weight = k - idx_floor
    return sorted_vals[idx_floor] + weight * (sorted_vals[idx_ceil] - sorted_vals[idx_floor])


def run_benchmark(
    cases: Optional[List[EvalCase]] = None,
    mode: Optional[str] = None,
    top_k: int = 5,
    limit: Optional[int] = None,
    category: Optional[str] = None,
    orchestrator: Optional[RAGOrchestrator] = None,
    verbose: bool = False,
    trace: bool = False,
) -> BenchmarkReport:
    """
    Execute evaluation benchmark against the active RAG orchestrator.
    Measures retrieval quality (MRR, Recall, Precision), citation accuracy,
    abstention reliability, latency distribution percentiles, and routing rates.
    """
    selected_cases = cases or EVALUATION_DATASET

    if category:
        selected_cases = [c for c in selected_cases if c.category.lower() == category.lower()]

    if limit is not None and limit > 0:
        selected_cases = selected_cases[:limit]

    if not selected_cases:
        logger.warning("No evaluation cases selected for benchmark run.")
        return BenchmarkReport()

    # Reuse singleton instances from SystemRegistry
    if orchestrator is None:
        embedder = SystemRegistry.get_embedder()
        qdrant = SystemRegistry.get_qdrant()
        bm25 = SystemRegistry.get_bm25()
        doc_store = SystemRegistry.get_doc_store()
        reranker = SystemRegistry.get_reranker()

        if qdrant.count() == 0 and bm25.count() > 0:
            from ..components.chunking.models import Chunk
            loaded_chunks = [Chunk.from_dict(c) for c in bm25._all_chunks]
            vecs = embedder.embed_texts([c.text for c in loaded_chunks])
            qdrant.upsert_chunks(loaded_chunks, vecs)

        orchestrator = RAGOrchestrator(
            embedder=embedder,
            qdrant_indexer=qdrant,
            bm25_indexer=bm25,
            document_store=doc_store,
            reranker=reranker,
        )

    report = BenchmarkReport(total_cases=len(selected_cases))

    reciprocal_ranks: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []
    abstention_correct = 0
    citation_valid_count = 0
    answerable_count = 0
    latencies: List[float] = []

    fast_count = 0
    balanced_count = 0
    deep_count = 0
    escalated_count = 0
    reranked_count = 0
    multi_query_count = 0
    parent_exp_count = 0
    abstained_total = 0

    cat_recalls: Dict[str, List[float]] = {}
    cat_precisions: Dict[str, List[float]] = {}
    cat_mrr: Dict[str, List[float]] = {}
    cat_abstention: Dict[str, List[int]] = {}
    cat_citations: Dict[str, List[int]] = {}
    cat_latencies: Dict[str, List[float]] = {}

    forced_mode = mode if mode and mode != "adaptive" else None

    for idx, case in enumerate(selected_cases, start=1):
        t0 = time.perf_counter()
        resp = orchestrator.query(case.query, mode=forced_mode)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        ret_meta = resp.retrieval
        if ret_meta.execution_mode == "fast":
            fast_count += 1
        elif ret_meta.execution_mode == "deep":
            deep_count += 1
        else:
            balanced_count += 1

        if ret_meta.escalated:
            escalated_count += 1
        if ret_meta.reranked:
            reranked_count += 1
        if ret_meta.multi_query_used:
            multi_query_count += 1
        if ret_meta.parent_expanded:
            parent_exp_count += 1
        if resp.abstained:
            abstained_total += 1

        cat = case.category
        if cat not in cat_recalls:
            cat_recalls[cat] = []
            cat_precisions[cat] = []
            cat_mrr[cat] = []
            cat_abstention[cat] = []
            cat_citations[cat] = []
            cat_latencies[cat] = []

        cat_latencies[cat].append(latency)

        case_detail: Dict[str, Any] = {
            "query_id": case.query_id,
            "category": case.category,
            "query": case.query,
            "expected_answerable": case.answerable,
            "actual_abstained": resp.abstained,
            "confidence": resp.confidence,
            "execution_mode": resp.execution_mode,
            "stages": ret_meta.cascade_stages_executed,
            "escalated": ret_meta.escalated,
            "latency_ms": latency,
            "sources_count": len(resp.sources),
        }

        # Abstention accuracy
        is_abs_correct = (case.answerable and not resp.abstained) or (not case.answerable and resp.abstained)
        if is_abs_correct:
            abstention_correct += 1
            cat_abstention[cat].append(1)
        else:
            cat_abstention[cat].append(0)

        # Retrieval accuracy for answerable cases
        if case.answerable:
            answerable_count += 1
            retrieved_texts = [
                (s.get("text", "") or s.get("snippet", "")).lower()
                for s in resp.sources
            ]

            first_rank = None
            hits_count = 0
            for rank, chunk_text in enumerate(retrieved_texts, start=1):
                if any(kw.lower() in chunk_text for kw in case.expected_doc_keywords):
                    hits_count += 1
                    if first_rank is None:
                        first_rank = rank

            rr = 1.0 / first_rank if first_rank is not None else 0.0
            rec = 1.0 if hits_count > 0 else 0.0
            prec = hits_count / max(1, len(retrieved_texts))

            reciprocal_ranks.append(rr)
            recalls.append(rec)
            precisions.append(prec)

            cat_mrr[cat].append(rr)
            cat_recalls[cat].append(rec)
            cat_precisions[cat].append(prec)

            # Citation validation
            val = resp.citation_validation
            is_cit_valid = val.get("is_valid", False) and len(val.get("invalid_citations", [])) == 0
            if is_cit_valid:
                citation_valid_count += 1
                cat_citations[cat].append(1)
            else:
                cat_citations[cat].append(0)

        report.details.append(case_detail)

        if trace:
            print("-" * 65)
            print(f"QUERY [{case.query_id}] : {case.query}")
            print(f"ROUTE      : {ret_meta.execution_mode.upper()} | STAGES: {ret_meta.cascade_stages_executed}")
            print(f"ESCALATED  : {ret_meta.escalated} | RERANKED: {ret_meta.reranked}")
            print(f"LATENCY    : {latency:.1f}ms | ABSTAINED: {resp.abstained}")
            if resp.sources:
                first_src = resp.sources[0]
                print(f"EVIDENCE #1: {first_src.get('filename')} ({first_src.get('snippet')[:70]}...)")
            print(f"ANSWER     : {resp.answer[:100]}...")
        elif verbose:
            status_str = "ABSTAINED" if resp.abstained else "ANSWERED"
            print(f"[{idx:03d}/{len(selected_cases):03d}] {case.query_id:12s} | {case.category:12s} | {status_str:9s} | {latency:6.1f}ms | {case.query[:45]}")

    # Aggregate global metrics
    n = max(1, len(selected_cases))
    report.mrr = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))
    report.recall_at_k = sum(recalls) / max(1, len(recalls))
    report.precision_at_k = sum(precisions) / max(1, len(precisions))
    report.abstention_accuracy = abstention_correct / n
    report.citation_accuracy = citation_valid_count / max(1, answerable_count)

    # Routing rates
    report.fast_path_rate = fast_count / n
    report.balanced_path_rate = balanced_count / n
    report.deep_path_rate = deep_count / n
    report.escalation_rate = escalated_count / n
    report.rerank_rate = reranked_count / n
    report.multi_query_rate = multi_query_count / n
    report.parent_expansion_rate = parent_exp_count / n
    report.abstention_rate = abstained_total / n

    # Aggregate latency distribution
    if latencies:
        report.avg_latency_ms = sum(latencies) / len(latencies)
        report.p50_latency_ms = _compute_percentile(latencies, 50.0)
        report.p90_latency_ms = _compute_percentile(latencies, 90.0)
        report.p95_latency_ms = _compute_percentile(latencies, 95.0)
        report.p99_latency_ms = _compute_percentile(latencies, 99.0)
        report.min_latency_ms = min(latencies)
        report.max_latency_ms = max(latencies)

    # Per-category metrics
    for cat in cat_latencies:
        cat_rec = sum(cat_recalls[cat]) / max(1, len(cat_recalls[cat])) if cat_recalls[cat] else 1.0
        cat_prec = sum(cat_precisions[cat]) / max(1, len(cat_precisions[cat])) if cat_precisions[cat] else 1.0
        cat_m = sum(cat_mrr[cat]) / max(1, len(cat_mrr[cat])) if cat_mrr[cat] else 1.0
        cat_abs = sum(cat_abstention[cat]) / max(1, len(cat_abstention[cat]))
        cat_cit = sum(cat_citations[cat]) / max(1, len(cat_citations[cat])) if cat_citations[cat] else 1.0
        cat_lats = cat_latencies[cat]

        report.category_metrics[cat] = {
            "cases": len(cat_lats),
            "recall_at_k": round(cat_rec, 4),
            "precision_at_k": round(cat_prec, 4),
            "mrr": round(cat_m, 4),
            "abstention_accuracy": round(cat_abs, 4),
            "citation_accuracy": round(cat_cit, 4),
            "avg_latency_ms": round(sum(cat_lats) / len(cat_lats), 1),
            "p50_latency_ms": round(_compute_percentile(cat_lats, 50.0), 1),
            "p95_latency_ms": round(_compute_percentile(cat_lats, 95.0), 1),
        }

    return report


def print_report(rep: BenchmarkReport, mode_label: str = "adaptive") -> None:
    """Print clean, structured benchmark results table."""
    print("-" * 75)
    print(f"ResearchLens — Benchmark Evaluation Report [Mode: {mode_label.upper()}]")
    print("-" * 75)
    print(f"Total Test Cases      : {rep.total_cases}")
    print(f"MRR (Mean Reciprocal) : {rep.mrr:.4f}")
    print(f"Recall@K              : {rep.recall_at_k:.4f}")
    print(f"Precision@K           : {rep.precision_at_k:.4f}")
    print(f"Abstention Accuracy   : {rep.abstention_accuracy * 100:.1f}%")
    print(f"Citation Accuracy     : {rep.citation_accuracy * 100:.1f}%")
    print("-" * 75)
    print("Routing Distribution:")
    print(f"  Fast Path Rate      : {rep.fast_path_rate * 100:5.1f}%")
    print(f"  Balanced Path Rate  : {rep.balanced_path_rate * 100:5.1f}%")
    print(f"  Deep Path Rate      : {rep.deep_path_rate * 100:5.1f}%")
    print(f"  Escalation Rate     : {rep.escalation_rate * 100:5.1f}%")
    print(f"  Rerank Rate         : {rep.rerank_rate * 100:5.1f}%")
    print(f"  Multi-Query Rate    : {rep.multi_query_rate * 100:5.1f}%")
    print(f"  Parent Expansion    : {rep.parent_expansion_rate * 100:5.1f}%")
    print(f"  Abstention Rate     : {rep.abstention_rate * 100:5.1f}%")
    print("-" * 75)
    print("Latency Distribution (ms):")
    print(f"  Min   : {rep.min_latency_ms:6.1f} ms")
    print(f"  P50   : {rep.p50_latency_ms:6.1f} ms")
    print(f"  P90   : {rep.p90_latency_ms:6.1f} ms")
    print(f"  P95   : {rep.p95_latency_ms:6.1f} ms")
    print(f"  P99   : {rep.p99_latency_ms:6.1f} ms")
    print(f"  Mean  : {rep.avg_latency_ms:6.1f} ms")
    print(f"  Max   : {rep.max_latency_ms:6.1f} ms")
    print("-" * 75)

    if rep.category_metrics:
        print(f"{'Category':<15} | {'Cases':<5} | {'Recall':<6} | {'Prec':<6} | {'MRR':<6} | {'Citations':<9} | {'P50 (ms)':<8} | {'P95 (ms)':<8}")
        print("-" * 75)
        for cat, m in rep.category_metrics.items():
            print(
                f"{cat:<15} | {m['cases']:<5} | {m['recall_at_k']:<6.2f} | "
                f"{m['precision_at_k']:<6.2f} | {m['mrr']:<6.2f} | "
                f"{m['citation_accuracy'] * 100:<8.1f}% | {m['p50_latency_ms']:<8.1f} | {m['p95_latency_ms']:<8.1f}"
            )
        print("-" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchLens — Production Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "deep", "adaptive"],
        default="adaptive",
        help="Pipeline execution mode (default: adaptive)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation cases")
    parser.add_argument("--category", type=str, default=None, help="Filter cases by category")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieval depth")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON benchmark output")
    parser.add_argument("--verbose", action="store_true", help="Print per-case execution logs")
    parser.add_argument("--trace", action="store_true", help="Print per-query end-to-end trace view")

    args = parser.parse_args()

    rep = run_benchmark(
        mode=args.mode,
        top_k=args.top_k,
        limit=args.limit,
        category=args.category,
        verbose=args.verbose,
        trace=args.trace,
    )

    print_report(rep, mode_label=args.mode)

    if args.output:
        out_dict = asdict(rep)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Benchmark report saved to: {args.output}")


if __name__ == "__main__":
    main()
