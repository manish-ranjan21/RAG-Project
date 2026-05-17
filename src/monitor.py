import json
from datetime import datetime, timezone
from pathlib import Path
import config


class Monitor:
    def __init__(self, log_path: str = config.LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._session = {"queries": 0, "total_latency": 0.0, "low_confidence": 0}

    def log(self, query: str, answer: str, sources: list,
            retrieval_ms: float, generation_ms: float, avg_score: float):
        self._session["queries"] += 1
        self._session["total_latency"] += retrieval_ms + generation_ms
        if avg_score > 0.5:
            self._session["low_confidence"] += 1

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": answer[:300],
            "sources": sources,
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "avg_chunk_score": round(avg_score, 3),
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def session_stats(self) -> dict:
        q = self._session["queries"]
        return {
            "total_queries": q,
            "avg_latency_ms": round(self._session["total_latency"] / q, 1) if q else 0,
            "low_confidence_answers": self._session["low_confidence"],
        }
