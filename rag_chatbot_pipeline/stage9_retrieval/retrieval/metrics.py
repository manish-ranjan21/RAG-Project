"""
Persist per-query metrics to a JSONL file for later analysis.

One line per query. Easy to ingest into pandas, BigQuery, Datadog, etc.
In production, replace with CloudWatch Embedded Metric Format or
direct emit to your observability stack.

Example analysis:
    cat logs/query_log.jsonl | jq 'select(.error == null) | .total_duration_seconds' | sort -n | tail -10
    cat logs/query_log.jsonl | jq -s 'map(select(.cache_hit)) | length'
"""

import json
import threading
from pathlib import Path

from .logging_setup import QueryContext


class MetricsWriter:
    """Thread-safe JSONL writer for query metrics."""

    def __init__(self, metrics_file: str | None, query_log_file: str | None):
        self.metrics_file = Path(metrics_file) if metrics_file else None
        self.query_log_file = Path(query_log_file) if query_log_file else None
        self._lock = threading.Lock()

        for path in (self.metrics_file, self.query_log_file):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)

    def record_query(self, ctx: QueryContext):
        if not self.query_log_file:
            return
        line = json.dumps(ctx.to_dict())
        with self._lock:
            with open(self.query_log_file, "a") as f:
                f.write(line + "\n")

    def record_metric(self, name: str, value, tags: dict | None = None):
        if not self.metrics_file:
            return
        record = {"metric": name, "value": value, "tags": tags or {}}
        line = json.dumps(record)
        with self._lock:
            with open(self.metrics_file, "a") as f:
                f.write(line + "\n")
