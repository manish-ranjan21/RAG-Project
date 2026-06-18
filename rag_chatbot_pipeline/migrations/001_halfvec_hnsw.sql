-- =====================================================================
-- Migration 001: switch embeddings to halfvec(3072) + add HNSW index
--
-- WHY: a plain `vector` column only supports an ANN index up to 2000 dims,
-- so our 3072-dim OpenAI embeddings could not be HNSW-indexed and every
-- query fell back to an exact sequential scan. `halfvec` (16-bit floats)
-- supports HNSW up to 4000 dims, halves storage, and costs negligible
-- recall for cosine similarity.
--
-- Safe to run on an existing volume that already holds vector(3072) rows:
-- the USING clause casts the stored vectors in place. Idempotent-ish —
-- the index uses IF NOT EXISTS; re-running the ALTER on an already-halfvec
-- column is a no-op cast.
-- =====================================================================

ALTER TABLE chunks
    ALTER COLUMN embedding TYPE halfvec(3072)
    USING embedding::halfvec(3072);

-- Cosine HNSW index — matches the `<=>` operator used by the retrieval service.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding halfvec_cosine_ops);
