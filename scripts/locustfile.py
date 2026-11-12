"""Locust load test for Nexus Adaptive RAG API.

Example headless run:
    locust -f scripts/locustfile.py --headless -u 5 -r 1 -t 60s \
        --host http://localhost:8000 --csv results/load
"""
from __future__ import annotations

import random
from locust import HttpUser, between, task

QUERIES = [
    ("What is the Transformer architecture?", "adaptive"),
    ("Explain scaled dot-product attention.", "fast"),
    ("Compare dense retrieval and BM25 with RRF fusion.", "balanced"),
    ("How does conditional cross-encoder reranking reduce tail latency?", "deep"),
]


class AdaptiveRAGUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")

    @task(2)
    def search(self) -> None:
        q, _ = random.choice(QUERIES)
        self.client.post(
            "/search",
            json={"query": q, "top_k": 5, "strategy": "hybrid"},
            name="/search",
        )

    @task(5)
    def query(self) -> None:
        q, mode = random.choice(QUERIES)
        self.client.post(
            "/query",
            json={"query": q, "mode": mode},
            name=f"/query[{mode}]",
        )
