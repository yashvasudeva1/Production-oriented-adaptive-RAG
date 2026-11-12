from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


class JSONLogFormatter(logging.Formatter):
    """Structured single-line JSON formatter for API and request telemetry logs."""

    _EXTRA_FIELDS = (
        "event",
        "request_id",
        "method",
        "path",
        "status_code",
        "latency_ms",
        "routing_mode",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in self._EXTRA_FIELDS:
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_json_logging(level: str = "INFO") -> None:
    """Attach JSONLogFormatter to the root logger once."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if any(isinstance(getattr(h, "formatter", None), JSONLogFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    root.handlers = [handler]


class PrometheusMetrics:
    """Thread-safe in-memory Prometheus text exposition metrics."""

    _BUCKETS_MS = (10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Dict[Tuple[str, str, int], int] = {}
        self._routing_modes: Dict[str, int] = {"fast": 0, "balanced": 0, "deep": 0}
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self._BUCKETS_MS}
        self._latency_sum_ms: float = 0.0
        self._latency_count: int = 0

    def observe_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        routing_mode: Optional[str] = None,
    ) -> None:
        with self._lock:
            key = (method, path, status_code)
            self._requests[key] = self._requests.get(key, 0) + 1
            self._latency_sum_ms += latency_ms
            self._latency_count += 1
            for bound in self._BUCKETS_MS:
                if latency_ms <= bound:
                    self._bucket_counts[bound] += 1
            if routing_mode:
                mode_key = routing_mode.lower()
                self._routing_modes[mode_key] = self._routing_modes.get(mode_key, 0) + 1

    def render_prometheus(self, snapshot: Dict[str, int]) -> str:
        lines: List[str] = [
            "# HELP nexus_requests_total Total HTTP requests processed.",
            "# TYPE nexus_requests_total counter",
        ]
        with self._lock:
            for (method, path, code), count in sorted(self._requests.items()):
                lines.append(
                    f'nexus_requests_total{{method="{method}",endpoint="{path}",status="{code}"}} {count}'
                )
            lines.extend([
                "# HELP nexus_query_routing_total Queries executed by adaptive routing mode.",
                "# TYPE nexus_query_routing_total counter",
            ])
            for mode, count in sorted(self._routing_modes.items()):
                lines.append(f'nexus_query_routing_total{{mode="{mode}"}} {count}')

            lines.extend([
                "# HELP nexus_request_latency_ms Request latency distribution in milliseconds.",
                "# TYPE nexus_request_latency_ms histogram",
            ])
            for bound in self._BUCKETS_MS:
                lines.append(
                    f'nexus_request_latency_ms_bucket{{le="{bound:g}"}} {self._bucket_counts[bound]}'
                )
            lines.append(f'nexus_request_latency_ms_bucket{{le="+Inf"}} {self._latency_count}')
            lines.append(f"nexus_request_latency_ms_sum {self._latency_sum_ms:.3f}")
            lines.append(f"nexus_request_latency_ms_count {self._latency_count}")

        gauges = {
            "nexus_documents_total": ("Total documents registered.", snapshot["documents_total"]),
            "nexus_indexed_documents": ("Documents in INDEXED state.", snapshot["indexed_documents"]),
            "nexus_qdrant_points_count": ("Vectors stored in Qdrant.", snapshot["qdrant_points_count"]),
            "nexus_bm25_chunks_count": ("Chunks indexed in BM25.", snapshot["bm25_chunks_count"]),
            "nexus_active_jobs_count": ("Active background ingestion jobs.", snapshot["active_jobs_count"]),
        }
        for name, (help_text, value) in gauges.items():
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        return "\n".join(lines) + "\n"
