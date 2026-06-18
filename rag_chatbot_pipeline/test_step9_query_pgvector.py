"""
Stage 9 (test): query pgvector to verify retrieval quality.

This is the read side of the pipeline — it proves that the vectors loaded by
step8 actually retrieve relevant chunks for a natural-language question.

How it works:
1. Embed the question with the SAME model step7 used for the chunks
   (text-embedding-3-large, 3072-dim). Using any other model would compare
   vectors from different spaces — meaningless results or a dimension error.
2. Ask Postgres for the nearest chunks by cosine distance (the `<=>` operator,
   which pairs with the vector_cosine_ops we chose). Lower distance = closer,
   so cosine_similarity = 1 - distance (1.0 = identical, 0 = unrelated).
3. Print the hits so you can eyeball whether they're on-topic.

Usage:
    export PGPASSWORD="$POSTGRES_PASSWORD"
    python src/step9_query_pgvector.py "your question here"
    python src/step9_query_pgvector.py "your question" --k 10
"""

import argparse
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY (and anything else) from the pipeline-local .env
load_dotenv(Path(__file__).resolve().parent / ".env")

# MUST match the model step7 stored. Do not change without re-embedding.
EMBED_MODEL = "text-embedding-3-large"


def embed_question(text: str) -> list[float]:
    """Embed the query with the same model as the stored chunks."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return resp.data[0].embedding


def vector_literal(vec: list[float]) -> str:
    """Render a python list as a pgvector literal: [0.1,0.2,...]."""
    return "[" + ",".join(map(str, vec)) + "]"


def search(question: str, k: int = 5):
    qvec = vector_literal(embed_question(question))

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="ragchatbot",
        user="ragchatbot_loader",
        password=os.environ.get("PGPASSWORD", ""),
        sslmode="prefer",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    chunk_id,
                    section_heading,
                    1 - (embedding <=> %s::vector) AS cosine_similarity,
                    left(text_for_embedding, 220) AS preview
                FROM chunks
                ORDER BY embedding <=> %s::vector   -- nearest first
                LIMIT %s
                """,
                (qvec, qvec, k),
            )
            return cur.fetchall()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pgvector retrieval for a question")
    parser.add_argument("question", help="The user question to retrieve chunks for")
    parser.add_argument("--k", type=int, default=5, help="How many chunks to return")
    args = parser.parse_args()

    rows = search(args.question, args.k)

    print("\n" + "=" * 70)
    print(f"QUESTION: {args.question}")
    print("=" * 70)
    if not rows:
        print("No chunks found — is the table empty? Run step8 first.")
    for i, (chunk_id, heading, sim, preview) in enumerate(rows, 1):
        print(f"\n[{i}] similarity={sim:.4f}  heading={heading!r}")
        print(f"    chunk_id: {chunk_id}")
        print(f"    {preview.strip()!r}")
