"""
Structured logging for the retrieval service.

In production, you want one log line per query with all the relevant fields
(query_id, latency, retrieved_count, cache_hit, etc.) so you can parse logs
into a metrics pipeline. Two formats supported:

- text: human-readable for CLI exploration
- json: machine-parseable for log aggregation (Splunk/DataDog/CloudWatch)
"""

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class QueryContext:
    """All the per-query state we want to log together."""

    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    query_text: str | None = None
    started_at: float = field(default_factory=time.monotonic)

    embed_duration_seconds: float | None = None
    sql_duration_seconds: float | None = None
    rerank_duration_seconds: float | None = None
    total_duration_seconds: float | None = None

    cache_hit: bool = False
    results_returned: int = 0
    top_similarity: float | None = None
    candidates_considered: int = 0

    embedding_model: str | None = None
    estimated_cost_usd: float = 0.0

    error: str | None = None

    def finalize(self):
        if self.total_duration_seconds is None:
            self.total_duration_seconds = time.monotonic() - self.started_at

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "user_id": self.user_id,
            "query_text": self.query_text,
            "embed_duration_seconds": self.embed_duration_seconds,
            "sql_duration_seconds": self.sql_duration_seconds,
            "rerank_duration_seconds": self.rerank_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "cache_hit": self.cache_hit,
            "results_returned": self.results_returned,
            "top_similarity": self.top_similarity,
            "candidates_considered": self.candidates_considered,
            "embedding_model": self.embedding_model,
            "estimated_cost_usd": self.estimated_cost_usd,
            "error": self.error,
            "timestamp": time.time(),
        }


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            payload.update(record.extra_data)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO", format_type: str = "text"):
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    if format_type == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(handler)


@contextmanager
def log_query(query_text: str, user_id: str | None = None):
    """Context manager that creates a QueryContext and logs on exit."""
    ctx = QueryContext(query_text=query_text, user_id=user_id)
    log = logging.getLogger("retrieval.query")

    try:
        yield ctx
    except Exception as e:
        ctx.error = str(e)
        log.error(f"Query {ctx.query_id} failed: {e}", extra={"extra_data": ctx.to_dict()})
        raise
    finally:
        ctx.finalize()
        if ctx.error is None:
            log.info(
                f"Query {ctx.query_id} ok "
                f"({ctx.total_duration_seconds * 1000:.0f}ms, "
                f"{ctx.results_returned} hits)",
                extra={"extra_data": ctx.to_dict()},
            )
