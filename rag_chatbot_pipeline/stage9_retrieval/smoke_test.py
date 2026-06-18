"""Smoke test verifying the modules work without needing a DB or API key."""

import os

os.environ["PGPASSWORD"] = "smoke_test"
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["EMBEDDING_MODEL"] = "mock-embedder"
os.environ["EMBEDDING_DIMENSION"] = "128"

from retrieval.cache import EmbeddingCache
from retrieval.config import load_config
from retrieval.embedder import build_embedder
from retrieval.logging_setup import log_query, setup_logging

print("=" * 60)
print("SMOKE TEST: Stage 9 Modules")
print("=" * 60)

setup_logging("INFO", "text")
config = load_config()
print(f"✓ Config loaded ({config.environment} env)")
print(f"  Embedding: {config.embedding.provider}/{config.embedding.model_name}")
print(f"  Dimension: {config.embedding.dimension}")
print(f"  Pool: {config.db.pool_min_conn}-{config.db.pool_max_conn} conns")

cache = EmbeddingCache(max_entries=5, ttl_seconds=60)
embedder = build_embedder(config.embedding, cache)
print(f"✓ Embedder built: {embedder.model_name}")

queries = [
    "what is machine learning",
    "how to evaluate a model",
    "what is machine learning",
    "feature engineering techniques",
    "what is machine learning",
]

print("\nEmbedding queries (note cache hits):")
for q in queries:
    vec, was_cached = embedder.embed(q)
    print(f"  '{q[:35]}' → {len(vec)} dims, {'CACHE HIT' if was_cached else 'fresh'}")

stats = cache.stats()
print(
    f"\n✓ Cache stats: hit_rate={stats['hit_rate'] * 100:.0f}%, "
    f"size={stats['size']}/{stats['max_entries']}"
)

print("\n✓ Query context tracking:")
with log_query("test query", user_id="smoke_user") as ctx:
    vec, was_cached = embedder.embed("test query")
    ctx.embed_duration_seconds = 0.05
    ctx.cache_hit = was_cached
    ctx.results_returned = 5
    ctx.top_similarity = 0.87
print(f"  Tracked: id={ctx.query_id[:8]}... duration={ctx.total_duration_seconds * 1000:.0f}ms")

print("\n" + "=" * 60)
print("All smoke tests passed")
print("=" * 60)
print("\nNext steps:")
print("1. Install: pip install psycopg2-binary pgvector openai")
print("2. Copy .env.example to .env and fill in your values")
print("3. Run: python -m retrieval.cli --health-check")
print("4. Query: python -m retrieval.cli 'your question'")
