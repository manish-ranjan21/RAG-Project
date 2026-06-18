"""
Thread-safe LRU cache with TTL for embedding results.

Same query often gets asked multiple times. Caching the embedding result
saves OpenAI calls and latency. In production, replace this with Redis;
the EmbeddingCache abstraction lets you swap implementations cleanly.

Why this matters: a single OpenAI embedding call costs ~$0.0002 and takes
~200ms. Caching common queries (40% hit rate is typical) cuts both.
"""
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Optional


class EmbeddingCache:
    """LRU cache with TTL. Thread-safe."""
    
    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    @staticmethod
    def _make_key(text: str, model: str) -> str:
        """Hash the text + model name as the cache key."""
        combined = f"{model}::{text}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]
    
    def get(self, text: str, model: str) -> Optional[list[float]]:
        key = self._make_key(text, model)
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            value, expires_at = self._cache[key]
            
            if time.monotonic() > expires_at:
                del self._cache[key]
                self.misses += 1
                return None
            
            self._cache.move_to_end(key)
            self.hits += 1
            return value
    
    def set(self, text: str, model: str, embedding: list[float]):
        key = self._make_key(text, model)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (embedding, time.monotonic() + self.ttl_seconds)
                return
            
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)
                self.evictions += 1
            
            self._cache[key] = (embedding, time.monotonic() + self.ttl_seconds)
    
    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
            }
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
