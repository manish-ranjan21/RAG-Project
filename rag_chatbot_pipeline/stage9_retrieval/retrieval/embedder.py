"""
Embeddings module - pluggable provider with retries and caching.

Same Embedder ABC pattern as Stage 7 (consistency matters). Adding caching
on top so the retrieval path is fast for repeated queries.

The retry logic matters: OpenAI rate-limits, network blips happen. Without
retries a single 503 takes down the user's query. With retries it adds 1-2s
of latency for the rare failure but the query still completes.
"""

import logging
import time
from abc import ABC, abstractmethod

from .cache import EmbeddingCache
from .config import EmbeddingConfig

log = logging.getLogger("retrieval.embedder")


class EmbedderError(Exception):
    pass


class Embedder(ABC):
    """Abstract interface - same shape as Stage 7's Embedder."""

    model_name: str
    model_version: str
    dimension: int
    cost_per_1k_tokens: float

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns the vector."""
        pass

    def estimate_cost(self, tokens: int) -> float:
        return (tokens / 1000) * self.cost_per_1k_tokens


class OpenAIEmbedder(Embedder):
    """Production OpenAI embedder with retries and timeouts."""

    def __init__(self, config: EmbeddingConfig):
        if not config.api_key:
            raise EmbedderError("OpenAI API key required")

        try:
            from openai import OpenAI
        except ImportError as e:
            raise EmbedderError("openai package not installed. pip install openai") from e

        self.client = OpenAI(api_key=config.api_key, timeout=config.timeout_seconds)
        self.model_name = config.model_name
        self.model_version = config.model_version
        self.dimension = config.dimension
        self.cost_per_1k_tokens = config.cost_per_1k_tokens
        self.max_retries = config.max_retries

    def embed(self, text: str) -> list[float]:
        text = text.strip()
        if not text:
            raise EmbedderError("Cannot embed empty text")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(input=[text], model=self.model_name)
                vector = response.data[0].embedding

                if len(vector) != self.dimension:
                    raise EmbedderError(
                        f"Expected dimension {self.dimension}, got {len(vector)}. "
                        f"Model may have changed."
                    )
                return vector
            except EmbedderError:
                raise
            except Exception as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    break
                wait = (2**attempt) * 0.5
                log.warning(f"Embed failed (attempt {attempt + 1}), retry in {wait:.1f}s: {e}")
                time.sleep(wait)

        raise EmbedderError(f"Failed after {self.max_retries} attempts: {last_error}")


class MockEmbedder(Embedder):
    """
    Deterministic mock for local testing without API keys.
    Same text always produces same vector. Different texts produce different vectors.
    """

    def __init__(self, config: EmbeddingConfig):
        self.model_name = "mock-embedder"
        self.model_version = "v1"
        self.dimension = config.dimension
        self.cost_per_1k_tokens = 0.0

    def embed(self, text: str) -> list[float]:
        import hashlib

        import numpy as np

        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.normal(0, 1, self.dimension)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()


class CachedEmbedder:
    """
    Decorator: wraps any Embedder with a cache.
    Transparent - same interface, just fewer API calls.
    """

    def __init__(self, underlying: Embedder, cache: EmbeddingCache):
        self.underlying = underlying
        self.cache = cache
        self.model_name = underlying.model_name
        self.model_version = underlying.model_version
        self.dimension = underlying.dimension
        self.cost_per_1k_tokens = underlying.cost_per_1k_tokens

    def embed(self, text: str) -> tuple[list[float], bool]:
        """Returns (vector, was_cache_hit)."""
        cached = self.cache.get(text, self.model_name)
        if cached is not None:
            return cached, True

        vector = self.underlying.embed(text)
        self.cache.set(text, self.model_name, vector)
        return vector, False

    def estimate_cost(self, tokens: int) -> float:
        return self.underlying.estimate_cost(tokens)


def build_embedder(config: EmbeddingConfig, cache: EmbeddingCache | None = None) -> CachedEmbedder:
    """Factory: build the right embedder for the configured provider."""
    if config.provider == "openai":
        underlying = OpenAIEmbedder(config)
    elif config.provider == "mock":
        underlying = MockEmbedder(config)
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")

    if cache is None:
        cache = EmbeddingCache(max_entries=1, ttl_seconds=1)

    return CachedEmbedder(underlying, cache)
