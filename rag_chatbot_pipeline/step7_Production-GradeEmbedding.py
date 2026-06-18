"""
Stage 7: Production-Grade Embedding

What "production-grade" means in practice:
1. BATCHED API calls - reduces cost and latency
2. RETRIES with exponential backoff - APIs fail sometimes
3. IDEMPOTENCY - skip chunks already embedded with same model version
4. VERSIONING - store model name + version with each vector
5. CONTENT HASHING - detect chunks that changed and need re-embedding
6. PARALLEL processing - use multiple workers for throughput
7. CHECKPOINTING - resume after failures without re-doing work
8. METRICS - track tokens, cost, latency per batch
9. WRITE PATH - both to S3 (durability) and vector DB (query)

Three embedder options demonstrated:
- BedrockCohereEmbedder: production banking choice (stays in AWS)
- OpenAIEmbedder: most common, requires Azure OpenAI for banking
- LocalBGEEmbedder: free, self-hosted, good for dev
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import Path as _Path

from dotenv import load_dotenv

# Read OPENAI_API_KEY (and friends) from the pipeline-local .env into os.environ.
load_dotenv(_Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class EmbeddingMetrics:
    """Track what happened during embedding for monitoring and cost attribution."""

    chunks_processed: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0
    total_tokens: int = 0
    total_api_calls: int = 0
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    errors: list = field(default_factory=list)


def compute_content_hash(text: str) -> str:
    """Deterministic hash of chunk text - used for idempotency."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Embedder(ABC):
    """Abstract interface so we can swap embedding providers cleanly."""

    model_name: str = ""
    model_version: str = ""
    dimension: int = 0
    cost_per_1k_tokens: float = 0.0
    max_batch_size: int = 100

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Must handle retries internally."""
        pass

    def estimate_cost(self, total_tokens: int) -> float:
        return (total_tokens / 1000) * self.cost_per_1k_tokens


class MockEmbedder(Embedder):
    """
    Deterministic mock embedder for demo purposes.
    In production, replace with BedrockCohereEmbedder, AzureOpenAIEmbedder, etc.
    Produces vectors based on text hash - same text always gives same vector.
    """

    model_name = "mock-bge-large-en-v1.5"
    model_version = "v1.5"
    dimension = 1024
    cost_per_1k_tokens = 0.0
    max_batch_size = 100

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.normal(0, 1, self.dimension)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec.tolist())
        return vectors


class BedrockCohereEmbedder(Embedder):
    """
    Production banking choice - uses Cohere via AWS Bedrock.
    Data stays in your AWS account. IAM authentication, audit logged.
    """

    model_name = "cohere.embed-english-v3"
    model_version = "v3"
    dimension = 1024
    cost_per_1k_tokens = 0.10 / 1000
    max_batch_size = 96

    def __init__(self, region: str = "us-east-1"):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        max_retries = 5
        backoff_base = 1.0

        for attempt in range(max_retries):
            try:
                body = json.dumps(
                    {"texts": texts, "input_type": "search_document", "truncate": "END"}
                )

                response = self.client.invoke_model(
                    modelId=self.model_name,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )

                result = json.loads(response["body"].read())
                return result["embeddings"]

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = backoff_base * (2**attempt)
                log.warning(
                    f"Bedrock call failed (attempt {attempt + 1}), retrying in {wait}s: {e}"
                )
                time.sleep(wait)


class AzureOpenAIEmbedder(Embedder):
    """
    Azure OpenAI - data residency compliant version of OpenAI.
    Required for banking if you want OpenAI models.
    """

    model_name = "text-embedding-3-large"
    model_version = "v3"
    dimension = 3072
    cost_per_1k_tokens = 0.13 / 1000
    max_batch_size = 100

    def __init__(self, model: str = "text-embedding-3-large"):
        import os

        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        max_retries = 5
        backoff_base = 1.0

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(input=texts, model=self.deployment)
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = backoff_base * (2**attempt)
                log.warning(f"Azure OpenAI failed (attempt {attempt + 1}), retry in {wait}s: {e}")
                time.sleep(wait)


class OpenAIEmbedder(Embedder):
    """
    Standard OpenAI API (api.openai.com).
    Simpler than Azure - only needs an API key, no endpoint or deployment.
    """

    model_name = "text-embedding-3-large"
    model_version = "v3"
    dimension = 3072
    cost_per_1k_tokens = 0.13 / 1000
    max_batch_size = 100

    def __init__(self, model: str = "text-embedding-3-large"):
        import os

        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        max_retries = 5
        backoff_base = 1.0

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(input=texts, model=self.model_name)
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = backoff_base * (2**attempt)
                log.warning(f"OpenAI failed (attempt {attempt + 1}), retry in {wait}s: {e}")
                time.sleep(wait)


class CheckpointStore:
    """
    Tracks which chunks have been embedded with which model version.
    In production this is DynamoDB or Postgres. Here we use a JSON file.
    Enables idempotency and incremental re-runs.
    """

    def __init__(self, checkpoint_path: str):
        self.path = Path(checkpoint_path)
        self.state = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2)

    def is_already_embedded(self, chunk_id: str, content_hash: str, model_version: str) -> bool:
        """Skip chunks that haven't changed and were embedded with same model."""
        record = self.state.get(chunk_id)
        if not record:
            return False
        return record["content_hash"] == content_hash and record["model_version"] == model_version

    def mark_embedded(self, chunk_id: str, content_hash: str, model_version: str, vector_path: str):
        self.state[chunk_id] = {
            "content_hash": content_hash,
            "model_version": model_version,
            "embedded_at": time.time(),
            "vector_path": vector_path,
        }


def chunks_iterator(chunks_path: str) -> Iterator[dict]:
    """Stream chunks from JSONL file without loading all into memory."""
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def batch_chunks(chunks: Iterator[dict], batch_size: int) -> Iterator[list[dict]]:
    """Group chunks into batches."""
    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_pipeline(
    chunks_path: str,
    output_path: str,
    checkpoint_path: str,
    embedder: Embedder,
    batch_size: int = 50,
    force_reembed: bool = False,
) -> EmbeddingMetrics:
    """
    Main embedding orchestration.
    Reads chunks, batches them, calls embedder, writes vectors to output.
    Skips already-embedded chunks unless force_reembed=True.
    """
    metrics = EmbeddingMetrics()
    checkpoint = CheckpointStore(checkpoint_path)
    start_time = time.time()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_log_path = str(output.parent / "runs.jsonl")

    # The embeddings file is the source of truth, keyed by chunk_id. Load it so we
    # can preserve already-embedded vectors and UPSERT new ones, instead of blindly
    # appending (which would duplicate records on every re-run). In real production
    # this keyed store is a vector DB; here it's an in-memory dict + atomic rewrite.
    vector_store: dict[str, dict] = {}
    if output.exists():
        with open(output) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    vector_store[rec["chunk_id"]] = rec

    def flush_vectors():
        """Atomically rewrite the embeddings file from the keyed store (no dupes)."""
        tmp = output.with_suffix(output.suffix + ".tmp")
        with open(tmp, "w") as f:
            for rec in vector_store.values():
                f.write(json.dumps(rec) + "\n")
        tmp.replace(output)  # atomic rename on the same filesystem

    all_chunks = list(chunks_iterator(chunks_path))
    total_chunks = len(all_chunks)
    log.info(f"Found {total_chunks} chunks to process")

    chunks_to_embed = []
    skipped_chunks = []

    for chunk in all_chunks:
        content_hash = compute_content_hash(chunk["text_for_embedding"])
        chunk["_content_hash"] = content_hash

        if not force_reembed and checkpoint.is_already_embedded(
            chunk["chunk_id"], content_hash, embedder.model_version
        ):
            skipped_chunks.append(chunk)
            metrics.chunks_skipped += 1
        else:
            chunks_to_embed.append(chunk)

    log.info(
        f"Will embed {len(chunks_to_embed)} chunks "
        f"({len(skipped_chunks)} skipped as already embedded)"
    )

    if not chunks_to_embed:
        log.info("Nothing to embed. All chunks already processed.")
        metrics.total_duration_seconds = time.time() - start_time
        append_run_log(metrics, run_log_path)
        return metrics

    batch_iter = batch_chunks(iter(chunks_to_embed), batch_size)

    for batch_idx, batch in enumerate(batch_iter):
        texts = [c["text_for_embedding"] for c in batch]
        batch_tokens = sum(c.get("token_count", 0) for c in batch)

        try:
            batch_start = time.time()
            vectors = embedder.embed_batch(texts)
            batch_duration = time.time() - batch_start

            metrics.total_api_calls += 1
            metrics.total_tokens += batch_tokens
            metrics.total_cost_usd += embedder.estimate_cost(batch_tokens)

            for chunk, vector in zip(batch, vectors, strict=False):
                embedded_record = {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "embedding": vector,
                    "embedding_model": embedder.model_name,
                    "embedding_model_version": embedder.model_version,
                    "embedding_dimension": embedder.dimension,
                    "embedded_at": time.time(),
                    "content_hash": chunk["_content_hash"],
                    "text_for_embedding": chunk["text_for_embedding"],
                    "section_path": chunk.get("section_path"),
                    "page_range": chunk.get("page_range"),
                    "token_count": chunk.get("token_count"),
                    "metadata": chunk.get("metadata", {}),
                }

                # Upsert by chunk_id: replaces a stale vector, never duplicates it.
                vector_store[chunk["chunk_id"]] = embedded_record

                checkpoint.mark_embedded(
                    chunk_id=chunk["chunk_id"],
                    content_hash=chunk["_content_hash"],
                    model_version=embedder.model_version,
                    vector_path=str(output_path),
                )

                metrics.chunks_processed += 1

            # Persist file first, then the checkpoint, so the checkpoint never
            # claims a chunk the file doesn't yet contain (crash-consistency).
            flush_vectors()
            checkpoint.save()

            log.info(
                f"Batch {batch_idx + 1}: embedded {len(batch)} chunks, "
                f"{batch_tokens} tokens, "
                f"{batch_duration:.2f}s, "
                f"${embedder.estimate_cost(batch_tokens):.5f}"
            )

        except Exception as e:
            log.error(f"Batch {batch_idx + 1} failed permanently: {e}")
            metrics.chunks_failed += len(batch)
            metrics.errors.append(
                {
                    "batch_idx": batch_idx,
                    "chunk_ids": [c["chunk_id"] for c in batch],
                    "error": str(e),
                }
            )

    metrics.total_duration_seconds = time.time() - start_time
    append_run_log(metrics, run_log_path)
    return metrics


def append_run_log(metrics: EmbeddingMetrics, run_log_path: str) -> dict:
    """
    Append one line per pipeline invocation to an audit log (JSONL).
    Unlike metrics.json (overwritten each run), this is append-only, so it answers
    "how many times has step 7 run, and how much did each run process vs. skip?"
    """
    p = Path(run_log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prior_runs = 0
    if p.exists():
        with open(p) as f:
            prior_runs = sum(1 for line in f if line.strip())
    record = {
        "run_number": prior_runs + 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "chunks_processed": metrics.chunks_processed,
        "chunks_skipped": metrics.chunks_skipped,
        "chunks_failed": metrics.chunks_failed,
        "total_cost_usd": round(metrics.total_cost_usd, 6),
        "total_duration_seconds": round(metrics.total_duration_seconds, 2),
    }
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")
    log.info(
        f"Run #{record['run_number']}: processed={record['chunks_processed']}, "
        f"skipped={record['chunks_skipped']}, failed={record['chunks_failed']}"
    )
    return record


def write_metrics(metrics: EmbeddingMetrics, path: str):
    """Save metrics for monitoring and cost attribution."""
    metrics_dict = {
        "chunks_processed": metrics.chunks_processed,
        "chunks_skipped": metrics.chunks_skipped,
        "chunks_failed": metrics.chunks_failed,
        "total_tokens": metrics.total_tokens,
        "total_api_calls": metrics.total_api_calls,
        "total_cost_usd": round(metrics.total_cost_usd, 6),
        "total_duration_seconds": round(metrics.total_duration_seconds, 2),
        "avg_tokens_per_chunk": (
            round(metrics.total_tokens / metrics.chunks_processed, 1)
            if metrics.chunks_processed > 0
            else 0
        ),
        "throughput_chunks_per_second": (
            round(metrics.chunks_processed / metrics.total_duration_seconds, 1)
            if metrics.total_duration_seconds > 0
            else 0
        ),
        "errors": metrics.errors,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    return metrics_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 7: production-grade embedding (incremental + idempotent)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing embeddings + checkpoint for a clean rebuild. "
        "Demo / disaster-recovery only — NOT something a production run does by default.",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Re-embed every chunk even if unchanged (e.g. after an embedding-model upgrade).",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    base_dir = (
        Path(__file__).resolve().parent
    )  # self-contained: data/ lives in this pipeline folder
    output_path = base_dir / "data" / "output"
    chunks_path = output_path / "ebook_chunks.jsonl"
    embeddings_path = output_path / "embeddings" / "ebook_embeddings.jsonl"
    checkpoint_path = output_path / "embeddings" / ".checkpoint.json"
    metrics_path = output_path / "embeddings" / "metrics.json"

    # Production default: PERSIST state across runs and embed only new/changed chunks.
    # A clean wipe is opt-in (--reset), never automatic — otherwise every run would
    # pay to re-embed work it already did.
    if args.reset:
        log.info("--reset: clearing existing embeddings + checkpoint for a clean rebuild")
        Path(embeddings_path).unlink(missing_ok=True)
        Path(checkpoint_path).unlink(missing_ok=True)

    embedder = OpenAIEmbedder()  # MockEmbedder()

    log.info(f"Starting embedding with {embedder.model_name} (dim={embedder.dimension})")
    log.info(f"Cost rate: ${embedder.cost_per_1k_tokens * 1000:.4f} per 1M tokens")

    metrics = embed_pipeline(
        chunks_path=chunks_path,
        output_path=embeddings_path,
        checkpoint_path=checkpoint_path,
        embedder=embedder,
        batch_size=args.batch_size,
        force_reembed=args.force_reembed,
    )

    metrics_dict = write_metrics(metrics, metrics_path)

    print("\n" + "=" * 60)
    print("STAGE 7: EMBEDDING COMPLETE")
    print("=" * 60)
    for k, v in metrics_dict.items():
        if k != "errors":
            print(f"  {k}: {v}")

    print(f"\nEmbeddings written: {embeddings_path}")
    print(f"Checkpoint state: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")

    print("\n" + "=" * 60)
    print("IDEMPOTENCY CHECK (re-run reuses the checkpoint, embeds nothing new)")
    print("=" * 60)

    metrics2 = embed_pipeline(
        chunks_path=chunks_path,
        output_path=embeddings_path,
        checkpoint_path=checkpoint_path,
        embedder=embedder,
        batch_size=args.batch_size,
        force_reembed=False,
    )

    print(f"  Second run skipped: {metrics2.chunks_skipped}")
    print(f"  Second run embedded: {metrics2.chunks_processed}")
    print(f"  Second run cost: ${metrics2.total_cost_usd:.6f}")
