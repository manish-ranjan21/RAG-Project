"""
Service bootstrap - ties everything together.

Single entry point that:
1. Loads config
2. Sets up logging
3. Initializes the connection pool
4. Runs the corpus health check (fail fast if misconfigured)
5. Builds the embedder with caching
6. Returns a fully wired RetrievalService

Use this in your CLI, batch eval, or future FastAPI app.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from .cache import EmbeddingCache
from .config import AppConfig, load_config
from .database import ConnectionPool, CorpusHealthCheck
from .embedder import build_embedder
from .logging_setup import setup_logging
from .metrics import MetricsWriter
from .reranker import build_reranker
from .service import RetrievalService


@dataclass
class Application:
    """Fully wired application - all the pieces a CLI/service/batch job needs."""

    config: AppConfig
    pool: ConnectionPool
    cache: EmbeddingCache
    service: RetrievalService
    metrics: MetricsWriter

    def shutdown(self):
        self.pool.close()


def bootstrap(env_file: Path | None = None, skip_health_check: bool = False) -> Application:
    """Build and validate the full application stack."""
    config = load_config(env_file)

    setup_logging(config.observability.log_level, config.observability.log_format)
    log = logging.getLogger("retrieval.bootstrap")

    log.info(f"Starting retrieval service ({config.environment})")
    log.info(f"DB: {config.db.connection_string(hide_password=True)}")
    log.info(
        f"Embedding: {config.embedding.provider}/{config.embedding.model_name} "
        f"dim={config.embedding.dimension}"
    )

    pool = ConnectionPool(config.db)
    try:
        pool.initialize()
    except Exception as e:
        log.error(f"Failed to initialize pool: {e}")
        raise

    if not skip_health_check:
        health = CorpusHealthCheck(pool, config.embedding.model_name, config.embedding.dimension)
        result = health.run()
        if not result["ok"]:
            log.error(f"Corpus health check failed: {result['error']}")
            pool.close()
            raise RuntimeError(f"Health check failed: {result['error']}")
        log.info(
            f"Health check OK: {result['chunk_count']} chunks, "
            f"model={result['expected_model']}, "
            f"hnsw_index={result['has_hnsw_index']}"
        )

    cache = (
        EmbeddingCache(max_entries=config.cache.max_entries, ttl_seconds=config.cache.ttl_seconds)
        if config.cache.enabled
        else EmbeddingCache(max_entries=1, ttl_seconds=1)
    )

    embedder = build_embedder(config.embedding, cache)

    reranker = None
    if config.retrieval.enable_rerank:
        reranker = build_reranker(
            config.retrieval.rerank_provider,
            config.retrieval.rerank_model,
        )
        log.info(
            f"Reranker: {config.retrieval.rerank_provider} "
            f"({config.retrieval.rerank_model}, loads on first use)"
        )

    service = RetrievalService(pool, embedder, config.retrieval, reranker)

    metrics = MetricsWriter(config.observability.metrics_file, config.observability.query_log_file)

    log.info("Service ready")

    return Application(config=config, pool=pool, cache=cache, service=service, metrics=metrics)
