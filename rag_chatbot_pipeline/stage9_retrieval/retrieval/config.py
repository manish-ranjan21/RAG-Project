"""
Centralized configuration for the retrieval service.

Loads from environment variables (.env file in dev, real env in production).
Validates everything on startup so you fail fast, not deep in a query.

In production, secrets come from AWS Secrets Manager instead of .env;
this module would be the only place that changes.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    sslmode: str = "prefer"
    pool_min_conn: int = 2
    pool_max_conn: int = 10
    statement_timeout_ms: int = 5000

    def connection_string(self, hide_password: bool = False) -> str:
        pw = "***" if hide_password else self.password
        return (f"postgresql://{self.user}:{pw}@{self.host}:{self.port}"
                f"/{self.database}?sslmode={self.sslmode}")


@dataclass
class EmbeddingConfig:
    provider: str
    model_name: str
    model_version: str
    dimension: int
    api_key: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    cost_per_1k_tokens: float = 0.0


@dataclass
class CacheConfig:
    enabled: bool = True
    max_entries: int = 1000
    ttl_seconds: int = 3600


@dataclass
class RetrievalConfig:
    default_top_k: int = 5
    max_top_k: int = 50
    # How many candidates retrieval hands the reranker. Bigger = better recall
    # but cross-encoder cost grows linearly (on CPU). 20 ≈ 4x top_k is a good balance.
    candidates_for_rerank: int = 20
    hnsw_ef_search: int = 60
    hybrid_vector_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3
    default_mode: str = "hybrid"
    enable_rerank: bool = True
    rerank_provider: str = "cross_encoder"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_similarity_threshold: float = 0.0


@dataclass
class ObservabilityConfig:
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_file: Optional[str] = None
    query_log_file: Optional[str] = None


@dataclass
class AppConfig:
    db: DatabaseConfig
    embedding: EmbeddingConfig
    cache: CacheConfig
    retrieval: RetrievalConfig
    observability: ObservabilityConfig
    environment: str = "development"

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.embedding.dimension <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {self.embedding.dimension}")
        if self.retrieval.default_top_k > self.retrieval.max_top_k:
            raise ValueError("default_top_k cannot exceed max_top_k")
        if abs((self.retrieval.hybrid_vector_weight + 
                self.retrieval.hybrid_keyword_weight) - 1.0) > 0.001:
            raise ValueError("hybrid weights must sum to 1.0")
        if self.environment == "production" and not self.embedding.api_key:
            raise ValueError("API key required in production")


def _load_dotenv(path: Path):
    """Minimal .env loader - avoids depending on python-dotenv."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _get_env(key: str, default=None, required: bool = False, cast=str):
    value = os.environ.get(key, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Required environment variable not set: {key}")
    if value is None:
        return None
    try:
        if cast is bool:
            return str(value).lower() in ("true", "1", "yes", "on")
        return cast(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse {key}={value} as {cast.__name__}: {e}")


def load_config(env_file: Optional[Path] = None) -> AppConfig:
    """Load config from environment with validation."""
    if env_file is None:
        env_file = Path.cwd() / ".env"
    _load_dotenv(env_file)
    
    db = DatabaseConfig(
        host=_get_env("PGHOST", "localhost"),
        port=_get_env("PGPORT", 5432, cast=int),
        database=_get_env("PGDATABASE", "ragchatbot"),
        user=_get_env("PGUSER", "ragchatbot_loader"),
        password=_get_env("PGPASSWORD", required=True),
        sslmode=_get_env("PGSSLMODE", "prefer"),
        pool_min_conn=_get_env("PG_POOL_MIN", 2, cast=int),
        pool_max_conn=_get_env("PG_POOL_MAX", 10, cast=int),
        statement_timeout_ms=_get_env("PG_STATEMENT_TIMEOUT_MS", 5000, cast=int),
    )
    
    embedding = EmbeddingConfig(
        provider=_get_env("EMBEDDING_PROVIDER", "openai"),
        model_name=_get_env("EMBEDDING_MODEL", "text-embedding-3-large"),
        model_version=_get_env("EMBEDDING_MODEL_VERSION", "v3"),
        dimension=_get_env("EMBEDDING_DIMENSION", 3072, cast=int),
        api_key=_get_env("OPENAI_API_KEY"),
        timeout_seconds=_get_env("EMBEDDING_TIMEOUT_SECONDS", 30, cast=int),
        max_retries=_get_env("EMBEDDING_MAX_RETRIES", 3, cast=int),
        cost_per_1k_tokens=_get_env("EMBEDDING_COST_PER_1K", 0.00013, cast=float),
    )
    
    cache = CacheConfig(
        enabled=_get_env("CACHE_ENABLED", True, cast=bool),
        max_entries=_get_env("CACHE_MAX_ENTRIES", 1000, cast=int),
        ttl_seconds=_get_env("CACHE_TTL_SECONDS", 3600, cast=int),
    )
    
    retrieval = RetrievalConfig(
        default_top_k=_get_env("RETRIEVAL_DEFAULT_K", 5, cast=int),
        max_top_k=_get_env("RETRIEVAL_MAX_K", 50, cast=int),
        candidates_for_rerank=_get_env("RETRIEVAL_RERANK_CANDIDATES", 50, cast=int),
        hnsw_ef_search=_get_env("RETRIEVAL_HNSW_EF", 60, cast=int),
        hybrid_vector_weight=_get_env("HYBRID_VECTOR_WEIGHT", 0.7, cast=float),
        hybrid_keyword_weight=_get_env("HYBRID_KEYWORD_WEIGHT", 0.3, cast=float),
        default_mode=_get_env("RETRIEVAL_DEFAULT_MODE", "hybrid"),
        enable_rerank=_get_env("ENABLE_RERANK", True, cast=bool),
        rerank_provider=_get_env("RERANK_PROVIDER", "cross_encoder"),
        rerank_model=_get_env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        min_similarity_threshold=_get_env("MIN_SIMILARITY_THRESHOLD", 0.0, cast=float),
    )
    
    observability = ObservabilityConfig(
        log_level=_get_env("LOG_LEVEL", "INFO"),
        log_format=_get_env("LOG_FORMAT", "text"),
        metrics_file=_get_env("METRICS_FILE", "logs/retrieval_metrics.jsonl"),
        query_log_file=_get_env("QUERY_LOG_FILE", "logs/query_log.jsonl"),
    )
    
    environment = _get_env("ENVIRONMENT", "development")
    
    return AppConfig(
        db=db, embedding=embedding, cache=cache, retrieval=retrieval,
        observability=observability, environment=environment
    )
