import json
from pathlib import Path

import pytest

from monitor import Monitor


@pytest.fixture
def monitor(tmp_path):
    log_file = tmp_path / "test_queries.jsonl"
    return Monitor(log_path=str(log_file))


def test_log_writes_to_file(monitor):
    monitor.log("What is AI?", "AI stands for...", ["book.pdf"], 40.0, 3000.0, 0.3)
    records = monitor.log_path.read_text().strip().splitlines()
    assert len(records) == 1

    entry = json.loads(records[0])
    assert entry["query"] == "What is AI?"
    assert entry["sources"] == ["book.pdf"]
    assert entry["retrieval_ms"] == 40.0
    assert entry["generation_ms"] == 3000.0
    assert entry["avg_chunk_score"] == 0.3
    assert "ts" in entry


def test_log_truncates_long_answer(monitor):
    long_answer = "x" * 500
    monitor.log("q?", long_answer, [], 10.0, 100.0, 0.2)
    entry = json.loads(monitor.log_path.read_text().strip())
    assert len(entry["answer"]) == 300


def test_multiple_logs_append(monitor):
    monitor.log("q1", "a1", [], 10.0, 100.0, 0.2)
    monitor.log("q2", "a2", [], 20.0, 200.0, 0.3)
    lines = monitor.log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_session_stats_empty(monitor):
    stats = monitor.session_stats()
    assert stats["total_queries"] == 0
    assert stats["avg_latency_ms"] == 0


def test_session_stats_after_queries(monitor):
    monitor.log("q1", "a1", [], 100.0, 900.0, 0.2)  # latency=1000ms, score ok
    monitor.log("q2", "a2", [], 100.0, 900.0, 0.8)  # latency=1000ms, low confidence

    stats = monitor.session_stats()
    assert stats["total_queries"] == 2
    assert stats["avg_latency_ms"] == 1000.0
    assert stats["low_confidence_answers"] == 1
