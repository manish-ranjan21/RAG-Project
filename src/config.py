from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
LLM_MODEL         = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
EMBED_MODEL       = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

RETRIEVAL_K           = int(os.getenv("RETRIEVAL_K", 3))
RELEVANCE_THRESHOLD   = float(os.getenv("RELEVANCE_THRESHOLD", 0.65))

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 750))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")
DB_PATH     = os.getenv("DB_PATH", "data/chroma")
LOG_PATH    = os.getenv("LOG_PATH", "data/rag_queries.jsonl")
