# Stage 9: Production-Grade Retrieval Service

A properly engineered retrieval layer for the RagChatbot RAG pipeline. Designed
to run on your local laptop while using the same patterns you'd deploy
to production.

## What's in the box

```
retrieval/
├── config.py         # Centralized config with validation
├── logging_setup.py  # Structured logging + per-query context
├── cache.py          # Thread-safe LRU + TTL embedding cache
├── embedder.py       # Pluggable embedding providers with retries
├── database.py       # Connection pooling + corpus health check
├── service.py        # The actual search logic (vector/keyword/hybrid)
├── metrics.py        # Persistent query metrics (JSONL)
├── bootstrap.py      # Wires everything together
└── cli.py            # Command-line entry point
```

## Setup

### 1. Install dependencies

```bash
pip install psycopg2-binary pgvector openai numpy
# For the default cross-encoder reranker (pulls in torch). Skip if you set
# RERANK_PROVIDER=heuristic or always pass --no-rerank.
pip install sentence-transformers
```

### 2. Configure

Copy the example env file and fill in your values:

```bash
cp .env.example .env
# Edit .env with your DB password and OpenAI API key
```

The most important values to match Stage 8:
- `PGDATABASE`, `PGUSER`, `PGPASSWORD` (where you loaded the chunks)
- `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` (must match Stage 7)

### 3. Test the connection

```bash
python -m retrieval.cli --health-check
```

If healthy, you'll see a summary with chunk count, model, dimensions, and
whether the HNSW index exists.

## Usage

### Basic queries

```bash
# Pure vector search (like your original Stage 9)
python -m retrieval.cli "what is feature engineering?"

# Get more results
python -m retrieval.cli "what is feature engineering?" --k 10

# JSON output for piping
python -m retrieval.cli "what is feature engineering?" --json | jq '.results[].chunk_id'
```

### Search modes

```bash
# Pure vector (semantic)
python -m retrieval.cli "data leakage in ML" --mode vector

# Pure keyword (full-text search)
python -m retrieval.cli "FINRA Rule 2111" --mode keyword

# Hybrid (recommended for most queries)
python -m retrieval.cli "data leakage in ML" --mode hybrid
```

### Reranking

Hybrid search + cross-encoder reranking is the **default** — end users never pick
a mode. Retrieval casts a wide net (`RETRIEVAL_RERANK_CANDIDATES` candidates) and a
local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores them for true
relevance, keeping the top `--k`. The model loads lazily on first use.

```bash
# Default: hybrid + rerank, no flags needed
python -m retrieval.cli "how to evaluate a model"

# Disable rerank for lower latency (skips the cross-encoder)
python -m retrieval.cli "how to evaluate a model" --no-rerank

# Use the lexical heuristic reranker instead (no torch, fully offline)
RERANK_PROVIDER=heuristic python -m retrieval.cli "how to evaluate a model"
```

### Permission filtering

```bash
# Only retrieve chunks classified as public
python -m retrieval.cli "anything" --access-class public

# Multiple access groups
python -m retrieval.cli "anything" --access-group advisors --access-group research

# Filter by document type
python -m retrieval.cli "anything" --doc-type policy --doc-type research_report
```

### Performance / debugging

```bash
# Show cache and pool stats
python -m retrieval.cli "query" --show-stats

# Disable cache for fair latency comparison
python -m retrieval.cli "query" --no-cache

# Tune HNSW recall (higher = better recall, slower)
python -m retrieval.cli "query" --ef-search 100

# Filter low-quality matches
python -m retrieval.cli "query" --min-similarity 0.3
```

### Test without API key

```bash
# Uses mock embedder, no OpenAI calls
python -m retrieval.cli "query" --mock
```

## Observability

Every query writes one JSON line to `logs/query_log.jsonl` with:
- query_id, query_text, user_id
- embed_duration_seconds, sql_duration_seconds, total_duration_seconds
- cache_hit, results_returned, top_similarity
- embedding_model, estimated_cost_usd
- error (if failed)

### Useful queries on the log

```bash
# P95 latency
cat logs/query_log.jsonl | jq '.total_duration_seconds' | sort -n | \
    awk 'BEGIN{c=0} {a[c++]=$1} END{print a[int(c*0.95)]}'

# Cache hit rate
cat logs/query_log.jsonl | jq -s '
  {total: length, hits: map(select(.cache_hit)) | length}
  | .hit_rate = (.hits / .total)
'

# Total cost today
cat logs/query_log.jsonl | jq -s 'map(.estimated_cost_usd) | add'

# Slowest queries
cat logs/query_log.jsonl | jq -s 'sort_by(.total_duration_seconds) | reverse | .[:5]'

# Failed queries
cat logs/query_log.jsonl | jq 'select(.error != null)'
```

## Using as a library

Don't want to use the CLI? Import the service:

```python
from retrieval.bootstrap import bootstrap
from retrieval.service import RetrievalRequest
from retrieval.logging_setup import log_query

app = bootstrap()

request = RetrievalRequest(
    query="what is feature engineering?",
    top_k=5,
    search_mode="hybrid",
    rerank=True
)

with log_query(request.query, user_id="analyst_123") as ctx:
    results = app.service.retrieve(request, ctx)

for r in results:
    print(f"{r.rank}. {r.chunk_id} (score={r.combined_score:.3f})")

app.shutdown()
```

## What's production-grade about this?

| Concern | What we do |
|---|---|
| Connection management | Thread-safe pool, statement timeouts, leases tracked |
| Embedding cost | LRU cache with TTL, avoids redundant OpenAI calls |
| Embedding failures | Exponential backoff, 3 retries, clear error messages |
| Permission enforcement | Required filter on every query, defaults are conservative |
| Configuration | Centralized, validated on startup, secrets from env/Secrets Manager |
| Observability | Per-query structured logs, persistent metrics, pool/cache stats |
| Schema drift | Pre-flight health check verifies model/dimension match |
| Search quality | Three modes, optional rerank, tunable HNSW parameters |
| Testability | Mock embedder for offline development |
| Error boundaries | One bad query never kills the service |

## What's NOT in here (intentionally)

- **FastAPI / HTTP layer**: this is a library + CLI. Add HTTP separately when needed.
- **GPU reranking / a reranker server**: the cross-encoder runs on CPU in-process.
  At scale, host it behind a batching endpoint (or use GPU) instead of loading
  per-process. A `heuristic` provider (lexical, no torch) is available as a fallback.
- **Redis cache**: in-process cache only. Swap CachingEmbedder when you scale out.
- **Distributed tracing**: structured logs only. Add OpenTelemetry when needed.
- **Auth / authz**: assumes the caller has already authenticated and resolved permissions.

These are deliberate omissions - YAGNI for local dev. Add them as you grow.

## What comes next

- **Stage 10**: Eval harness with golden query sets, recall@k measurement
- **Stage 11**: RAG loop (retrieval + LLM synthesis with citations)
- **Stage 12**: FastAPI service wrapping this for the chatbot frontend
