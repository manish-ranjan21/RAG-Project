-- =====================================================================
-- pgvector schema for the RAG pipeline (Stage 8 loader)
-- Loaded automatically by Postgres on first container start
-- (mounted into /docker-entrypoint-initdb.d).
--
-- Embedding dimension = 3072  (OpenAI text-embedding-3-large, what step7
-- actually produces). The column length MUST match the vectors exactly or
-- every insert is rejected. If you switch models, change vector(3072).
--
-- NOTE: pgvector's HNSW/IVFFlat indexes on a plain `vector` column only
-- support <= 2000 dims, so the 3072-dim embeddings are stored as
-- `halfvec(3072)` (16-bit floats), which HNSW supports up to 4000 dims.
-- See the HNSW index below (halfvec_cosine_ops).
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------
-- documents: one row per source document
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id                TEXT PRIMARY KEY,
    doc_type              TEXT,
    title                 TEXT,
    source_system         TEXT,
    source_path           TEXT,
    content_hash          TEXT,
    page_count            INTEGER,
    author                TEXT,
    publication_date      DATE,
    effective_date        DATE,
    expiry_date           DATE,
    access_classification TEXT DEFAULT 'internal',
    access_groups         TEXT[] DEFAULT '{}',
    pipeline_version      TEXT,
    metadata              JSONB DEFAULT '{}'::jsonb,
    is_deleted            BOOLEAN DEFAULT FALSE,
    last_ingested_at      TIMESTAMPTZ DEFAULT NOW(),
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------
-- ingestion_runs: one row per load run (observability / lineage)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id                  UUID PRIMARY KEY,
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    status                  TEXT,
    pipeline_version        TEXT,
    embedding_model         TEXT,
    embedding_model_version TEXT,
    source_file             TEXT,
    operator                TEXT,
    chunks_loaded           INTEGER DEFAULT 0,
    chunks_failed           INTEGER DEFAULT 0,
    error_summary           TEXT
);

-- ---------------------------------------------------------------------
-- chunks: the high-volume table with embeddings
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id                TEXT PRIMARY KEY,
    doc_id                  TEXT REFERENCES documents(doc_id),
    run_id                  UUID REFERENCES ingestion_runs(run_id),
    embedding               halfvec(3072),
    embedding_model         TEXT,
    embedding_model_version TEXT,
    text_for_embedding      TEXT,
    content_hash            TEXT,
    section_path            TEXT,
    section_heading         TEXT,
    section_level           INTEGER,
    page_range              INT4RANGE,
    chunk_index             INTEGER,
    chunk_type              TEXT,
    chunk_strategy          TEXT,
    token_count             INTEGER,
    access_classification   TEXT DEFAULT 'internal',
    access_groups           TEXT[] DEFAULT '{}',
    referenced_figures      JSONB DEFAULT '[]'::jsonb,
    referenced_tables       JSONB DEFAULT '[]'::jsonb,
    metadata                JSONB DEFAULT '{}'::jsonb,
    -- Full-text search vector, kept in sync with text_for_embedding by Postgres.
    -- Powers the keyword & hybrid retrieval modes (Stage 9).
    text_search             tsvector GENERATED ALWAYS AS (to_tsvector('english', text_for_embedding)) STORED,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_run_id ON chunks(run_id);

-- Full-text search for keyword / hybrid retrieval (Stage 9).
-- The ALTER backfills tables created before this column existed; on a fresh
-- install the column is already present (see CREATE TABLE above) so it's a no-op.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS text_search tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text_for_embedding)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_text_search ON chunks USING GIN (text_search);

-- ANN index for vector search. A plain `vector` column only supports an ANN
-- index up to 2000 dims, so the 3072-dim embeddings are stored as `halfvec`
-- (16-bit floats), which HNSW supports up to 4000 dims. Cosine ops to match
-- the `<=>` operator the retrieval service (Stage 9) uses. Per-query recall is
-- tuned with `SET LOCAL hnsw.ef_search = N`.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding halfvec_cosine_ops);
