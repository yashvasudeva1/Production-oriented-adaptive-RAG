"""Compute throughput (QPS), p50/p95 latency, and cost per 1,000 queries.

Supports:
  1. Reading a Locust *_stats.csv file (--locust-csv results/load_stats.csv)
  2. Reading a benchmark CSV containing a latency column (--latency-csv metadata_benchmark_results.csv)
  3. Running live sequential/concurrent probes against /query (--url http://127.0.0.1:8000/query)

Example:
    python scripts/benchmark_cost.py --url http://127.0.0.1:8000/query --requests 20 --vm-hourly-usd 0.036
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple


def percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def summarize(latencies_ms: List[float], total_wall_s: float, vm_hourly_usd: float) -> dict:
    ordered = sorted(latencies_ms)
    count = len(ordered)
    qps = (count / total_wall_s) if total_wall_s > 0 else 0.0
    p50 = percentile(ordered, 50)
    p95 = percentile(ordered, 95)
    cost_per_1k = (vm_hourly_usd / 3600.0) * (1000.0 / qps) if qps > 0 else 0.0
    return {
        "requests": count,
        "wall_seconds": round(total_wall_s, 3),
        "throughput_qps": round(qps, 3),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "vm_hourly_usd": vm_hourly_usd,
        "cost_per_1k_queries_usd": round(cost_per_1k, 6),
    }


def from_locust_stats_csv(path: Path, vm_hourly_usd: float) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    agg = next((r for r in rows if r.get("Name") == "Aggregated"), rows[-1])
    qps = float(agg.get("Requests/s") or 0.0)
    p50 = float(agg.get("50%") or agg.get("Median Response Time") or 0.0)
    p95 = float(agg.get("95%") or 0.0)
    count = int(float(agg.get("Request Count") or 0))
    cost_per_1k = (vm_hourly_usd / 3600.0) * (1000.0 / qps) if qps > 0 else 0.0
    return {
        "requests": count,
        "throughput_qps": round(qps, 3),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "vm_hourly_usd": vm_hourly_usd,
        "cost_per_1k_queries_usd": round(cost_per_1k, 6),
    }


def from_latency_csv(path: Path, column: str, vm_hourly_usd: float) -> dict:
    latencies: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if column in row and row[column]:
                latencies.append(float(row[column]))
    wall_s = sum(latencies) / 1000.0
    return summarize(latencies, wall_s, vm_hourly_usd)


def run_live_probe(url: str, requests_count: int, mode: str, vm_hourly_usd: float) -> dict:
    payload = json.dumps({"query": "What is the Transformer architecture?", "mode": mode}).encode("utf-8")
    latencies: List[float] = []
    t_wall0 = time.perf_counter()
    for _ in range(requests_count):
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    wall_s = time.perf_counter() - t_wall0
    return summarize(latencies, wall_s, vm_hourly_usd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute QPS, p50/p95 latency, and $/1k queries.")
    parser.add_argument("--locust-csv", type=Path, help="Path to Locust *_stats.csv file.")
    parser.add_argument("--latency-csv", type=Path, help="Path to CSV with per-query latency_ms.")
    parser.add_argument("--latency-col", default="latency_ms", help="Latency column name in --latency-csv.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/query", help="Live /query endpoint URL.")
    parser.add_argument("--requests", type=int, default=10, help="Number of live probe requests.")
    parser.add_argument("--mode", default="adaptive", help="Routing mode for live probe.")
    parser.add_argument("--vm-hourly-usd", type=float, default=0.036, help="VM hourly price in USD (e.g., 0.036 for 4GB Vultr, 0.0416 for AWS t3.medium).")
    args = parser.parse_args()

    if args.locust_csv:
        result = from_locust_stats_csv(args.locust_csv, args.vm_hourly_usd)
    elif args.latency_csv:
        result = from_latency_csv(args.latency_csv, args.latency_col, args.vm_hourly_usd)
    else:
        result = run_live_probe(args.url, args.requests, args.mode, args.vm_hourly_usd)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
