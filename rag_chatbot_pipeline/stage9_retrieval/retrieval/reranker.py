"""
Reranking module - pluggable cross-encoder with a heuristic fallback.

Why rerank at all?
The retrieval stage (vector/keyword/hybrid) optimizes for RECALL - cast a wide
net, get the ~40 candidates that *might* be relevant. But a bi-encoder embedding
compares the query and chunk independently, so it can't tell that a keyword-dense
index page is actually useless for the question.

A cross-encoder reads (query, chunk) TOGETHER and scores true relevance. We run
it only on the small candidate set from retrieval, then keep the top_k. This is
the standard "retrieve broad, rerank narrow" pattern.

Same ABC + factory shape as embedder.py for consistency. The CrossEncoder model
loads lazily on first use so --health-check / --mock / --no-rerank stay fast.
"""

import logging
import time
from abc import ABC, abstractmethod

log = logging.getLogger("retrieval.reranker")


class RerankerError(Exception):
    pass


class Reranker(ABC):
    """Abstract interface: score how relevant each text is to the query."""

    model_name: str

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, aligned by index. Higher = better."""
        pass


class CrossEncoderReranker(Reranker):
    """
    sentence-transformers cross-encoder. Default model cross-encoder/ms-marco-MiniLM-L-6-v2
    is small (~80MB), CPU-friendly (~50-150ms for 40 candidates), and trained on
    MS MARCO passage ranking - a strong general-purpose relevance reranker.

    The model is loaded lazily (first score() call) so importing/bootstrapping is cheap.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RerankerError(
                "sentence-transformers not installed. "
                "pip install sentence-transformers (pulls in torch)."
            ) from e
        t0 = time.monotonic()
        log.info(f"Loading cross-encoder '{self.model_name}' (first use)...")
        self._model = CrossEncoder(self.model_name)
        log.info(f"Cross-encoder loaded in {time.monotonic() - t0:.1f}s")

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        self._ensure_model()
        pairs = [(query, t) for t in texts]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


class TermOverlapReranker(Reranker):
    """
    Lightweight lexical fallback - the original heuristic. No model, fully offline.
    Used when sentence-transformers is unavailable or provider='heuristic'/'mock'.
    Scores by fraction of query terms present in the chunk.
    """

    def __init__(self):
        self.model_name = "term-overlap-heuristic"

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_terms = set(query.lower().split())
        denom = max(len(query_terms), 1)
        out = []
        for text in texts:
            overlap = len(query_terms & set(text.lower().split()))
            out.append(overlap / denom)
        return out


class FallbackReranker(Reranker):
    """
    Production wrapper: try the primary reranker (cross-encoder); if it fails,
    transparently degrade to the fallback (lexical heuristic) for THAT query and
    every query after, instead of dropping reranking entirely.

    Why latch after the first failure: a cross-encoder failure is almost always a
    load problem (missing torch, OOM, bad model name) that won't fix itself within
    the process. Retrying the broken model on every query would re-pay the slow,
    doomed load each time. So once it fails we stick with the fallback and log once.
    A transient inference blip is the rare exception and still degrades safely.
    """

    def __init__(self, primary: Reranker, fallback: Reranker):
        self.primary = primary
        self.fallback = fallback
        self.model_name = primary.model_name
        self._use_fallback = False

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not self._use_fallback:
            try:
                return self.primary.score(query, texts)
            except Exception as e:
                log.warning(
                    f"Primary reranker '{self.primary.model_name}' failed ({e}); "
                    f"degrading to fallback '{self.fallback.model_name}' for the rest "
                    f"of this process."
                )
                self._use_fallback = True
        return self.fallback.score(query, texts)


def build_reranker(provider: str, model_name: str) -> Reranker:
    """
    Factory. The cross-encoder is built lazily - the model only loads on the first
    rerank, so bootstrap / health-check / --no-rerank stay fast.

    For 'cross_encoder' we wrap it in a FallbackReranker so a model/load failure at
    query time degrades to the lexical heuristic (better than no reranking).
    """
    if provider in ("heuristic", "mock", "none"):
        return TermOverlapReranker()
    if provider == "cross_encoder":
        return FallbackReranker(CrossEncoderReranker(model_name), TermOverlapReranker())
    raise ValueError(f"Unknown reranker provider: {provider}")
