from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_api_health_and_ready():
    r_health = client.get("/health")
    assert r_health.status_code == 200
    data = r_health.json()
    assert "status" in data
    assert "document_count" in data

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "qdrant_points_count" in r_metrics.json()


def test_api_list_documents():
    r = client.get("/documents")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_api_search_endpoint():
    r = client.post(
        "/search",
        json={"query": "Transformer attention mechanism", "top_k": 3, "strategy": "hybrid"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "latency_ms" in data


def test_api_query_endpoint():
    r = client.post(
        "/query",
        json={"query": "What is the Transformer architecture?"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "confidence" in data
    assert "abstained" in data
    assert "retrieval" in data


def test_api_empty_query_validation():
    r = client.post("/query", json={"query": "   "})
    assert r.status_code == 400
