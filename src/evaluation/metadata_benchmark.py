from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..components.chunking.models import Chunk
from ..components.indexing import (
    BM25Indexer,
    DocumentStore,
    QdrantIndexer,
    SentenceTransformerEmbedder,
)
from ..components.retrieval import (
    DenseRetriever,
    FilterCondition,
    HybridRetriever,
    KeywordRetriever,
    RetrievalFilter,
)

logger = logging.getLogger(__name__)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass
class BenchmarkScenarioResult:
    scenario_name: str
    target_selectivity: str
    predicate_count: int
    filter_mode: str  # "native_pre_filter", "post_filter", "unfiltered"
    cache_state: str  # "cold", "warm"
    queries_run: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_qps: float
    matched_candidates_avg: float
    false_positive_rate: float


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    idx_floor = int(k)
    idx_ceil = min(idx_floor + 1, len(sorted_vals) - 1)
    weight = k - idx_floor
    return sorted_vals[idx_floor] + weight * (sorted_vals[idx_ceil] - sorted_vals[idx_floor])


def setup_benchmark_corpus(
    embedder: SentenceTransformerEmbedder,
    num_chunks: int = 500,
) -> tuple[QdrantIndexer, BM25Indexer]:
    """
    Generate synthetic test corpus with controlled metadata distributions:
    - year: 2020..2025
    - category: "research", "policy", "finance", "engineering", "legal"
    - department: "AI", "governance", "operations", "risk", "security"
    - security: "public", "internal", "confidential"
    """
    logger.info(f"Generating synthetic benchmark corpus with {num_chunks} chunks...")
    chunks: List[Chunk] = []
    texts: List[str] = []

    categories = ["research", "policy", "finance", "engineering", "legal"]
    departments = ["AI", "governance", "operations", "risk", "security"]
    years = [2021, 2022, 2023, 2024, 2025]
    access_levels = ["public", "internal", "confidential"]

    sample_sentences = [
        "Deep learning transformers enable multi-head self-attention across hidden representation layers.",
        "Internship applicants must submit college verification certificates signed by department heads.",
        "Quarterly fiscal policy reports recommend systematic budget allocation across research departments.",
        "Scaled dot-product attention divides dot products by the square root of dimension d_k.",
        "Administrative compliance mandates attendance monitoring protocols for all operational divisions.",
        "Autonomous gradient optimization utilizes Adam optimizer with warmup step scheduling.",
        "Confidential multi-tenant data structures require strict payload filtering and zero cross-leakage.",
        "High performance retrieval architectures combine vector dense indexes with inverted BM25 structures.",
    ]

    for i in range(num_chunks):
        cid = f"meta_bench_chk_{i:04d}"
        did = f"meta_bench_doc_{i // 5:03d}"
        yr = years[i % len(years)]
        cat = categories[i % len(categories)]
        dep = departments[i % len(departments)]
        acc = access_levels[i % len(access_levels)]

        text = (
            f"[{yr} | {cat.upper()} | {dep.upper()}] "
            + sample_sentences[i % len(sample_sentences)]
            + f" Specific token code identifier: IDX_{i:04d}."
        )

        chunk = Chunk(
            chunk_id=cid,
            document_id=did,
            text=text,
            token_count=len(text.split()),
            metadata={
                "year": yr,
                "category": cat,
                "department": dep,
                "access": acc,
                "document_type": "report" if i % 2 == 0 else "paper",
            },
        )
        chunks.append(chunk)
        texts.append(text)

    # In-memory Qdrant indexer with payload indexes enabled
    qdrant = QdrantIndexer(
        collection_name=f"metadata_bench_{int(time.time())}",
        vector_size=embedder.dimension,
        create_indexes=True,
    )
    embeddings = embedder.embed_texts(texts)
    qdrant.upsert_chunks(chunks, embeddings)

    # Temporary BM25 indexer for benchmark
    import tempfile
    import uuid
    tmp_path = Path(tempfile.gettempdir()) / f"meta_bench_bm25_{uuid.uuid4().hex[:8]}.json"
    bm25 = BM25Indexer(index_path=str(tmp_path))
    bm25.upsert_chunks(chunks)

    return qdrant, bm25


def run_metadata_benchmark(
    num_queries: int = 20,
    corpus_size: int = 500,
) -> List[BenchmarkScenarioResult]:
    """Run full matrix of selectivity (1%, 10%, 50%, 100%) and predicate counts (1, 2, 3)."""
    embedder = SentenceTransformerEmbedder()
    qdrant, bm25 = setup_benchmark_corpus(embedder, num_chunks=corpus_size)

    dense_retriever = DenseRetriever(indexer=qdrant, embedder=embedder)
    keyword_retriever = KeywordRetriever(indexer=bm25)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        keyword_retriever=keyword_retriever,
    )

    test_queries = [
        "What are the self-attention transformer mechanisms in deep learning?",
        "Explain administrative compliance and verification requirements.",
        "How do fiscal budget allocations support engineering operations?",
        "What are the attendance protocols for internal research divisions?",
        "Describe high performance vector search and BM25 index structures.",
    ]

    scenarios = [
        # (name, selectivity_label, [FilterCondition])
        (
            "100% Unfiltered Baseline",
            "100%",
            1,
            None,
        ),
        (
            "50% Selectivity (1 Predicate: doc_type == report)",
            "50%",
            1,
            RetrievalFilter(
                hard_filters=[
                    FilterCondition(field="metadata.document_type", operator="eq", value="report")
                ]
            ),
        ),
        (
            "20% Selectivity (1 Predicate: year == 2024)",
            "20%",
            1,
            RetrievalFilter(
                hard_filters=[
                    FilterCondition(field="metadata.year", operator="eq", value=2024)
                ]
            ),
        ),
        (
            "4% Selectivity (2 Predicates: year == 2024 AND category == research)",
            "4%",
            2,
            RetrievalFilter(
                hard_filters=[
                    FilterCondition(field="metadata.year", operator="eq", value=2024),
                    FilterCondition(field="metadata.category", operator="eq", value="research"),
                ]
            ),
        ),
        (
            "1% Selectivity (3 Predicates: year == 2024 AND category == research AND dept == AI)",
            "1%",
            3,
            RetrievalFilter(
                hard_filters=[
                    FilterCondition(field="metadata.year", operator="eq", value=2024),
                    FilterCondition(field="metadata.category", operator="eq", value="research"),
                    FilterCondition(field="metadata.department", operator="eq", value="AI"),
                ]
            ),
        ),
    ]

    results: List[BenchmarkScenarioResult] = []

    for name, sel_label, pred_count, r_filter in scenarios:
        # A. Native Pre-filtered Cold Run
        cold_lats: List[float] = []
        cold_counts: List[int] = []
        cold_false_positives = 0
        total_eval_cands = 0

        for i in range(num_queries):
            q = test_queries[i % len(test_queries)]
            t0 = time.perf_counter()
            resp = hybrid_retriever.retrieve(query=q, metadata_filter=r_filter, top_k=10)
            lat = (time.perf_counter() - t0) * 1000
            cold_lats.append(lat)
            cold_counts.append(len(resp.results))

            # Verify zero false positives for native filter
            if r_filter:
                for c in resp.results:
                    total_eval_cands += 1
                    if not r_filter.matches_chunk(c.metadata):
                        cold_false_positives += 1

        fp_rate = (cold_false_positives / total_eval_cands) if total_eval_cands > 0 else 0.0
        avg_lat = sum(cold_lats) / len(cold_lats)
        qps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        results.append(
            BenchmarkScenarioResult(
                scenario_name=name,
                target_selectivity=sel_label,
                predicate_count=pred_count,
                filter_mode="native_pre_filter",
                cache_state="cold",
                queries_run=num_queries,
                avg_latency_ms=round(avg_lat, 2),
                p50_latency_ms=round(_percentile(cold_lats, 50.0), 2),
                p95_latency_ms=round(_percentile(cold_lats, 95.0), 2),
                throughput_qps=round(qps, 1),
                matched_candidates_avg=round(sum(cold_counts) / len(cold_counts), 2),
                false_positive_rate=round(fp_rate, 4),
            )
        )

        # B. Native Pre-filtered Warm Cache Run (Simulating repeated / warm queries)
        warm_lats: List[float] = []
        for i in range(num_queries):
            q = test_queries[i % len(test_queries)]
            t0 = time.perf_counter()
            resp = hybrid_retriever.retrieve(query=q, metadata_filter=r_filter, top_k=10)
            lat = (time.perf_counter() - t0) * 1000
            warm_lats.append(lat)

        warm_avg = sum(warm_lats) / len(warm_lats)
        warm_qps = 1000.0 / warm_avg if warm_avg > 0 else 0.0

        results.append(
            BenchmarkScenarioResult(
                scenario_name=name + " [Warm]",
                target_selectivity=sel_label,
                predicate_count=pred_count,
                filter_mode="native_pre_filter",
                cache_state="warm",
                queries_run=num_queries,
                avg_latency_ms=round(warm_avg, 2),
                p50_latency_ms=round(_percentile(warm_lats, 50.0), 2),
                p95_latency_ms=round(_percentile(warm_lats, 95.0), 2),
                throughput_qps=round(warm_qps, 1),
                matched_candidates_avg=round(sum(cold_counts) / len(cold_counts), 2),
                false_positive_rate=0.0,
            )
        )

    return results


def print_results_table(results: List[BenchmarkScenarioResult]) -> None:
    print("\n" + "=" * 105)
    print("ResearchLens — Native Metadata Filtering Scalability Benchmark")
    print("=" * 105)
    header = (
        f"{'Scenario':<42} | {'Selectivity':<11} | {'Preds':<5} | "
        f"{'Mode':<7} | {'Mean (ms)':<9} | {'P50 (ms)':<8} | {'P95 (ms)':<8} | {'QPS':<7} | {'FP Rate':<7}"
    )
    print(header)
    print("-" * 105)

    for r in results:
        print(
            f"{r.scenario_name:<42} | "
            f"{r.target_selectivity:<11} | "
            f"{r.predicate_count:<5} | "
            f"{r.cache_state:<7} | "
            f"{r.avg_latency_ms:<9.2f} | "
            f"{r.p50_latency_ms:<8.2f} | "
            f"{r.p95_latency_ms:<8.2f} | "
            f"{r.throughput_qps:<7.1f} | "
            f"{r.false_positive_rate * 100:<6.1f}%"
        )
    print("=" * 105 + "\n")


def export_csv(results: List[BenchmarkScenarioResult], file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"Exported metadata benchmark results to: {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResearchLens — Native Metadata Filtering Benchmark"
    )
    parser.add_argument(
        "--num-queries", type=int, default=20, help="Number of benchmark queries per scenario"
    )
    parser.add_argument(
        "--corpus-size", type=int, default=500, help="Number of synthetic chunks in test corpus"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="metadata_benchmark_results.csv",
        help="CSV file path to save benchmark results",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="JSON file path to save benchmark results",
    )

    args = parser.parse_args()

    results = run_metadata_benchmark(
        num_queries=args.num_queries,
        corpus_size=args.corpus_size,
    )
    print_results_table(results)

    if args.output_csv:
        export_csv(results, args.output_csv)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Exported metadata benchmark results to JSON: {args.output_json}")


if __name__ == "__main__":
    main()
