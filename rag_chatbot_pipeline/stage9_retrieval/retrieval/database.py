"""
Database connection management with pooling and health checks.

Why pooling matters even locally:
- Per-query psycopg2.connect() takes ~50ms (TCP + auth + TLS)
- Per-query operations get noticeably slower under load
- A pool of 10 connections handles hundreds of QPS

The pool is a singleton initialized once at startup. ContextManager pattern
makes connection lease/return automatic and exception-safe.

In production, replace this with RDS Proxy (transparent pooling, IAM auth,
failover handling). The code that uses get_connection() doesn't change.
"""
import logging
import time
from contextlib import contextmanager
from typing import Optional

from .config import DatabaseConfig

log = logging.getLogger("retrieval.db")


class DatabaseError(Exception):
    pass


class ConnectionPool:
    """Wraps psycopg2 ThreadedConnectionPool with init validation and stats."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool = None
        self._leases = 0
        self._lease_total = 0
        self._lease_errors = 0
    
    def initialize(self):
        """Create the pool and verify connectivity. Call at startup."""
        try:
            from psycopg2 import pool
            from pgvector.psycopg2 import register_vector
        except ImportError as e:
            raise DatabaseError(
                f"Required packages not installed: {e}. "
                "pip install psycopg2-binary pgvector"
            )
        
        self._register_vector = register_vector
        
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.config.pool_min_conn,
                maxconn=self.config.pool_max_conn,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                sslmode=self.config.sslmode,
                connect_timeout=10,
            )
        except Exception as e:
            raise DatabaseError(f"Failed to create pool: {e}")
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                if cur.fetchone()[0] != 1:
                    raise DatabaseError("Pool init test query returned unexpected result")
        
        log.info(f"Pool initialized: {self.config.pool_min_conn}-{self.config.pool_max_conn} conns "
                f"to {self.config.host}:{self.config.port}/{self.config.database}")
    
    @contextmanager
    def get_connection(self):
        """
        Lease a connection from the pool. Returns to pool on exit, even on error.
        Sets statement_timeout to prevent runaway queries.
        """
        if self._pool is None:
            raise DatabaseError("Pool not initialized. Call initialize() first.")
        
        self._lease_total += 1
        self._leases += 1
        
        conn = None
        try:
            conn = self._pool.getconn()
            self._register_vector(conn)
            
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout TO {self.config.statement_timeout_ms}")
            
            yield conn
        except Exception as e:
            self._lease_errors += 1
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            self._leases -= 1
            if conn:
                self._pool.putconn(conn)
    
    def health_check(self) -> dict:
        """Return pool state for monitoring."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    healthy = cur.fetchone()[0] == 1
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "leases_active": self._leases,
                "leases_total": self._lease_total,
                "lease_errors": self._lease_errors,
            }
        
        return {
            "healthy": healthy,
            "leases_active": self._leases,
            "leases_total": self._lease_total,
            "lease_errors": self._lease_errors,
        }
    
    def close(self):
        if self._pool:
            self._pool.closeall()
            self._pool = None
            log.info("Pool closed")


class CorpusHealthCheck:
    """
    Pre-flight check that verifies the corpus is actually loaded and matches
    the configured embedding model. Catches the worst category of bugs
    (querying empty table, model mismatch) before user-facing failures.
    """
    
    def __init__(self, pool: ConnectionPool, expected_model: str, expected_dimension: int):
        self.pool = pool
        self.expected_model = expected_model
        self.expected_dimension = expected_dimension
    
    def run(self) -> dict:
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'chunks'
                    )
                """)
                if not cur.fetchone()[0]:
                    return {"ok": False, "error": "chunks table does not exist"}
                
                cur.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cur.fetchone()[0]
                if chunk_count == 0:
                    return {"ok": False, "error": "chunks table is empty"}
                
                cur.execute("""
                    SELECT embedding_model, COUNT(*) 
                    FROM chunks 
                    GROUP BY embedding_model
                """)
                models = dict(cur.fetchall())
                if self.expected_model not in models:
                    return {
                        "ok": False,
                        "error": (f"Expected model {self.expected_model} not found in chunks. "
                                 f"Found: {list(models.keys())}. "
                                 f"Re-embed or update EMBEDDING_MODEL config.")
                    }
                
                cur.execute("""
                    SELECT array_length(embedding::real[], 1)
                    FROM chunks 
                    WHERE embedding_model = %s 
                    LIMIT 1
                """, (self.expected_model,))
                actual_dim = cur.fetchone()[0]
                if actual_dim != self.expected_dimension:
                    return {
                        "ok": False,
                        "error": (f"Stored vectors have dimension {actual_dim} but config "
                                 f"expects {self.expected_dimension}")
                    }
                
                cur.execute("""
                    SELECT COUNT(*) FROM pg_indexes 
                    WHERE tablename = 'chunks' AND indexname LIKE '%hnsw%'
                """)
                hnsw_count = cur.fetchone()[0]
                
                return {
                    "ok": True,
                    "chunk_count": chunk_count,
                    "models": models,
                    "expected_model": self.expected_model,
                    "dimension": actual_dim,
                    "has_hnsw_index": hnsw_count > 0,
                }
