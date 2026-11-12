import json
import pytest

try:
    from fastapi.testclient import TestClient
    from src.api.app import app, state
    client = TestClient(app)
except Exception:
    pytest.skip("fastapi / TestClient unavailable in environment", allow_module_level=True)


def test_api_health_ready_and_prometheus_metrics():
    r_root = client.get("/")
    assert r_root.status_code == 200

    r_head = client.head("/health")
    assert r_head.status_code == 200

    r_health = client.get("/health")
    assert r_health.status_code == 200
    data = r_health.json()
    assert "status" in data
    assert "document_count" in data
    assert "qdrant_healthy" in data

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    assert r_ready.json()["status"] == "ready"

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "text/plain" in r_metrics.headers["content-type"]
    body = r_metrics.text
    assert "# TYPE nexus_requests_total counter" in body
    assert "# TYPE nexus_query_routing_total counter" in body
    assert "nexus_qdrant_points_count" in body

    r_metrics_json = client.get("/metrics/json")
    assert r_metrics_json.status_code == 200
    assert "qdrant_points_count" in r_metrics_json.json()


def test_api_ready_returns_503_when_qdrant_unhealthy(monkeypatch):
    monkeypatch.setattr(state.qdrant, "health_check", lambda: False)
    r = client.get("/ready")
    assert r.status_code == 503


def test_api_root_serves_frontend():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Nexus" in r.text


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


def test_api_query_and_stream_endpoints():
    r = client.post(
        "/query",
        json={"query": "What is the Transformer architecture?", "mode": "balanced"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "confidence" in data
    assert "abstained" in data
    assert "retrieval" in data
    assert data["execution_mode"] in ("fast", "balanced", "deep")

    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "What is the Transformer architecture?", "mode": "adaptive"},
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        event_types = [e["event"] for e in events]
        assert "retrieval" in event_types
        assert "delta" in event_types
        assert event_types[-1] == "done"


def test_api_input_validation():
    assert client.post("/query", json={"query": "   "}).status_code == 422
    assert client.post("/query", json={"query": "x" * 2001}).status_code == 422
    assert client.post("/query", json={"query": "valid", "mode": "ultra"}).status_code == 422
    assert client.post("/search", json={"query": "   "}).status_code == 422
    assert client.post("/search", json={"query": "valid", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "valid", "top_k": 51}).status_code == 422
    assert client.post("/search", json={"query": "valid", "strategy": "invalid"}).status_code == 422
