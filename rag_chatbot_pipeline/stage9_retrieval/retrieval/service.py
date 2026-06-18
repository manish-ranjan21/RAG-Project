"""
The main retrieval service.

Three retrieval modes:
- vector_search: pure semantic similarity (your original Stage 9)
- keyword_search: pure full-text search using tsvector
- hybrid_search: weighted combination of both

All modes support:
- Permission filtering (access_classification, access_groups)
- Date filtering (effective/expiry dates)
- Doc type filtering
- Minimum similarity threshold
- HNSW ef_search tuning per query

The interface is sync. To convert to async (for FastAPI), wrap calls with
asyncio.to_thread() - the SQL itself doesn't benefit from async.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import RetrievalConfig
from .database import ConnectionPool
from .embedder import CachedEmbedder
from .logging_setup import QueryContext
from .reranker import Reranker

log = logging.getLogger("retrieval.service")


@dataclass
class RetrievalResult:
    """One chunk returned from retrieval, with all metadata for downstream use."""
    chunk_id: str
    doc_id: str
    text: str
    section_heading: Optional[str]
    section_path: Optional[str]
    page_range: Optional[tuple]
    
    vector_similarity: Optional[float] = None
    keyword_score: Optional[float] = None
    combined_score: Optional[float] = None
    rerank_score: Optional[float] = None
    
    rank: int = 0
    embedding_model: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "section_heading": self.section_heading,
            "section_path": self.section_path,
            "page_range": list(self.page_range) if self.page_range else None,
            "vector_similarity": self.vector_similarity,
            "keyword_score": self.keyword_score,
            "combined_score": self.combined_score,
            "rerank_score": self.rerank_score,
            "rank": self.rank,
            "embedding_model": self.embedding_model,
        }


@dataclass
class RetrievalRequest:
    """The full set of options a caller can specify."""
    query: str
    top_k: int = 5
    
    access_classifications: list[str] = field(
        default_factory=lambda: ["public", "internal"]
    )
    access_groups: list[str] = field(default_factory=lambda: ["all_advisors"])
    doc_types: Optional[list[str]] = None
    
    search_mode: str = "hybrid"
    min_similarity: float = 0.0

    hnsw_ef_search: Optional[int] = None
    rerank: bool = True
    return_full_text: bool = True


def _vector_literal(vec: list[float]) -> str:
    """Render a vector as pgvector literal: [0.1,0.2,...]"""
    return "[" + ",".join(f"{x}" for x in vec) + "]"


def _parse_page_range(pg_range_str) -> Optional[tuple]:
    """Postgres int4range comes back as '[1,5)' - parse to (1, 4)."""
    if pg_range_str is None:
        return None
    if hasattr(pg_range_str, "lower") and hasattr(pg_range_str, "upper"):
        return (pg_range_str.lower, pg_range_str.upper - 1)
    return None


class RetrievalService:
    """Main retrieval service - one instance per process."""
    
    def __init__(
        self,
        pool: ConnectionPool,
        embedder: CachedEmbedder,
        config: RetrievalConfig,
        reranker: Optional[Reranker] = None
    ):
        self.pool = pool
        self.embedder = embedder
        self.config = config
        self.reranker = reranker
    
    def retrieve(
        self,
        request: RetrievalRequest,
        ctx: Optional[QueryContext] = None
    ) -> list[RetrievalResult]:
        """Main entry point - dispatches to the right search mode."""
        if request.top_k > self.config.max_top_k:
            raise ValueError(f"top_k {request.top_k} exceeds max {self.config.max_top_k}")
        
        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        # Retrieve-broad-then-rerank: when reranking, over-fetch a wider candidate
        # pool from the DB (recall), let the cross-encoder pick the best (precision),
        # then trim to top_k. Without rerank, fetch exactly top_k.
        do_rerank = request.rerank and self.reranker is not None
        fetch_limit = (max(request.top_k, self.config.candidates_for_rerank)
                       if do_rerank else request.top_k)

        if request.search_mode == "vector":
            results = self._vector_search(request, ctx, fetch_limit)
        elif request.search_mode == "keyword":
            results = self._keyword_search(request, ctx, fetch_limit)
        elif request.search_mode == "hybrid":
            results = self._hybrid_search(request, ctx, fetch_limit)
        else:
            raise ValueError(f"Unknown search mode: {request.search_mode}")

        if do_rerank and len(results) > 1:
            results = self._rerank(request.query, results, ctx)

        results = results[:request.top_k]

        for i, r in enumerate(results, 1):
            r.rank = i
        
        if ctx:
            ctx.results_returned = len(results)
            ctx.top_similarity = (results[0].vector_similarity or 
                                  results[0].combined_score) if results else None
        
        return results
    
    def _embed_query(self, query: str, ctx: Optional[QueryContext]) -> list[float]:
        t0 = time.monotonic()
        vector, was_cached = self.embedder.embed(query)
        duration = time.monotonic() - t0
        
        if ctx:
            ctx.embed_duration_seconds = duration
            ctx.cache_hit = was_cached
            ctx.embedding_model = self.embedder.model_name
            if not was_cached:
                approx_tokens = len(query) // 4
                ctx.estimated_cost_usd = self.embedder.estimate_cost(approx_tokens)
        
        return vector
    
    def _build_filters(self, request: RetrievalRequest):
        """Build the WHERE clause parameters for permission/filter clauses."""
        filters = ["c.access_classification = ANY(%s)", "c.access_groups && %s"]
        params = [request.access_classifications, request.access_groups]
        
        if request.doc_types:
            filters.append("d.doc_type = ANY(%s)")
            params.append(request.doc_types)
        
        filters.append("d.is_deleted = FALSE")
        filters.append("(d.effective_date IS NULL OR d.effective_date <= CURRENT_DATE)")
        filters.append("(d.expiry_date IS NULL OR d.expiry_date >= CURRENT_DATE)")
        
        return " AND ".join(filters), params
    
    def _vector_search(
        self, request: RetrievalRequest, ctx: Optional[QueryContext], limit: int
    ) -> list[RetrievalResult]:
        """Pure vector similarity search."""
        query_vec = self._embed_query(request.query, ctx)
        vec_literal = _vector_literal(query_vec)
        
        filter_clause, filter_params = self._build_filters(request)
        ef = request.hnsw_ef_search or self.config.hnsw_ef_search
        
        text_field = "c.text_for_embedding" if request.return_full_text else "left(c.text_for_embedding, 500)"
        
        sql = f"""
            SELECT
                c.chunk_id, c.doc_id,
                {text_field} AS text,
                c.section_heading, c.section_path, c.page_range,
                1 - (c.embedding <=> %s::halfvec) AS similarity,
                c.embedding_model
            FROM chunks c
            JOIN documents d USING (doc_id)
            WHERE {filter_clause}
              AND 1 - (c.embedding <=> %s::halfvec) >= %s
            ORDER BY c.embedding <=> %s::halfvec
            LIMIT %s
        """
        
        params = [vec_literal, *filter_params, vec_literal,
                  request.min_similarity, vec_literal, limit]
        
        t0 = time.monotonic()
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")
                cur.execute(sql, params)
                rows = cur.fetchall()
        sql_duration = time.monotonic() - t0
        
        if ctx:
            ctx.sql_duration_seconds = sql_duration
            ctx.candidates_considered = ef
        
        return [
            RetrievalResult(
                chunk_id=row[0], doc_id=row[1], text=row[2],
                section_heading=row[3], section_path=row[4],
                page_range=_parse_page_range(row[5]),
                vector_similarity=float(row[6]),
                embedding_model=row[7]
            )
            for row in rows
        ]
    
    def _keyword_search(
        self, request: RetrievalRequest, ctx: Optional[QueryContext], limit: int
    ) -> list[RetrievalResult]:
        """Full-text search using tsvector + GIN index."""
        filter_clause, filter_params = self._build_filters(request)
        text_field = "c.text_for_embedding" if request.return_full_text else "left(c.text_for_embedding, 500)"
        
        sql = f"""
            SELECT
                c.chunk_id, c.doc_id,
                {text_field} AS text,
                c.section_heading, c.section_path, c.page_range,
                ts_rank_cd(c.text_search, plainto_tsquery('english', %s)) AS kw_score,
                c.embedding_model
            FROM chunks c
            JOIN documents d USING (doc_id)
            WHERE {filter_clause}
              AND c.text_search @@ plainto_tsquery('english', %s)
            ORDER BY kw_score DESC
            LIMIT %s
        """
        
        params = [request.query, *filter_params, request.query, limit]
        
        t0 = time.monotonic()
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        if ctx:
            ctx.sql_duration_seconds = time.monotonic() - t0
        
        return [
            RetrievalResult(
                chunk_id=row[0], doc_id=row[1], text=row[2],
                section_heading=row[3], section_path=row[4],
                page_range=_parse_page_range(row[5]),
                keyword_score=float(row[6]),
                embedding_model=row[7]
            )
            for row in rows
        ]
    
    def _hybrid_search(
        self, request: RetrievalRequest, ctx: Optional[QueryContext], limit: int
    ) -> list[RetrievalResult]:
        """Combine vector similarity and keyword score with configured weights."""
        query_vec = self._embed_query(request.query, ctx)
        vec_literal = _vector_literal(query_vec)

        filter_clause, filter_params = self._build_filters(request)
        ef = request.hnsw_ef_search or self.config.hnsw_ef_search
        # CTE candidate pool must be at least as wide as the rows we return.
        candidates = max(self.config.candidates_for_rerank, limit)
        text_field = "c.text_for_embedding" if request.return_full_text else "left(c.text_for_embedding, 500)"
        
        sql = f"""
            WITH vec_hits AS (
                SELECT c.chunk_id,
                       1 - (c.embedding <=> %s::halfvec) AS vec_sim
                FROM chunks c
                JOIN documents d USING (doc_id)
                WHERE {filter_clause}
                ORDER BY c.embedding <=> %s::halfvec
                LIMIT %s
            ),
            kw_hits AS (
                SELECT c.chunk_id,
                       ts_rank_cd(c.text_search, plainto_tsquery('english', %s)) AS kw_score
                FROM chunks c
                JOIN documents d USING (doc_id)
                WHERE {filter_clause}
                  AND c.text_search @@ plainto_tsquery('english', %s)
                LIMIT %s
            ),
            normalized AS (
                SELECT
                    COALESCE(v.chunk_id, k.chunk_id) AS chunk_id,
                    COALESCE(v.vec_sim, 0) AS vec_sim,
                    COALESCE(k.kw_score / NULLIF((SELECT MAX(kw_score) FROM kw_hits), 0), 0) AS kw_norm
                FROM vec_hits v
                FULL OUTER JOIN kw_hits k USING (chunk_id)
            )
            SELECT
                c.chunk_id, c.doc_id,
                {text_field} AS text,
                c.section_heading, c.section_path, c.page_range,
                n.vec_sim, n.kw_norm,
                (%s * n.vec_sim + %s * n.kw_norm) AS combined,
                c.embedding_model
            FROM normalized n
            JOIN chunks c USING (chunk_id)
            WHERE (%s * n.vec_sim + %s * n.kw_norm) >= %s
            ORDER BY combined DESC
            LIMIT %s
        """
        
        params = [
            vec_literal, *filter_params, vec_literal, candidates,
            request.query, *filter_params, request.query, candidates,
            self.config.hybrid_vector_weight, self.config.hybrid_keyword_weight,
            self.config.hybrid_vector_weight, self.config.hybrid_keyword_weight,
            request.min_similarity, limit
        ]
        
        t0 = time.monotonic()
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")
                cur.execute(sql, params)
                rows = cur.fetchall()
        if ctx:
            ctx.sql_duration_seconds = time.monotonic() - t0
            ctx.candidates_considered = candidates
        
        return [
            RetrievalResult(
                chunk_id=row[0], doc_id=row[1], text=row[2],
                section_heading=row[3], section_path=row[4],
                page_range=_parse_page_range(row[5]),
                vector_similarity=float(row[6]) if row[6] else 0.0,
                keyword_score=float(row[7]) if row[7] else 0.0,
                combined_score=float(row[8]),
                embedding_model=row[9]
            )
            for row in rows
        ]
    
    def _rerank(
        self, query: str, results: list[RetrievalResult], ctx: Optional[QueryContext]
    ) -> list[RetrievalResult]:
        """
        Rerank candidates with the configured reranker (cross-encoder by default,
        term-overlap heuristic as fallback). The reranker reads (query, chunk_text)
        together, so it judges true relevance rather than embedding proximity.

        On any failure we keep the original retrieval order rather than dropping the
        query - one bad rerank should never lose results the user could have seen.
        """
        t0 = time.monotonic()

        try:
            scores = self.reranker.score(query, [r.text for r in results])
            for result, score in zip(results, scores):
                result.rerank_score = float(score)
            results.sort(key=lambda r: r.rerank_score if r.rerank_score is not None else float("-inf"),
                         reverse=True)
        except Exception as e:
            log.warning(f"Rerank failed ({e}); keeping retrieval order.")

        if ctx:
            ctx.rerank_duration_seconds = time.monotonic() - t0

        return results
