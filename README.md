# RAG Project

A local Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about AI/ML textbooks and get grounded, sourced answers — entirely offline using Ollama.

![CI](https://github.com/manish-ranjan21/RAG-Project/actions/workflows/ci.yml/badge.svg)

---

## How It Works

```
PDFs → Chunks → Embeddings → ChromaDB
                                  ↓
          Query → Guardrails → Retrieval → LLM → Answer
                                  ↓
                              Monitor (logs + stats)
```

1. **Ingest** — PDFs are loaded, split into 750-char chunks, embedded with `nomic-embed-text`, and stored in ChromaDB
2. **Guard** — Every query is validated: length, injection patterns, topic relevance, and retrieval quality
3. **Retrieve** — Top-k chunks are fetched via cosine similarity search
4. **Generate** — `llama3` answers using only the retrieved context
5. **Monitor** — Every query is timed, scored, and logged to a `.jsonl` file

---

## Project Structure

```
rag-project/
├── .github/workflows/ci.yml   ← GitHub Actions (runs tests on every push)
├── src/
│   ├── config.py              ← all settings loaded from .env
│   ├── loader.py              ← DocumentLoader (PDF loading + chunking)
│   ├── vector_store.py        ← VectorStore (ChromaDB)
│   ├── guardrails.py          ← input/relevance/injection guardrails
│   ├── monitor.py             ← query logging + session stats
│   ├── rag_chain.py           ← RAGChain (retrieval + generation)
│   ├── ingest.py              ← build the vector store from PDFs
│   ├── app.py                 ← Streamlit web UI
│   └── coding.py              ← terminal chat entry point
├── tests/
│   ├── conftest.py
│   ├── test_guardrails.py
│   ├── test_monitor.py
│   └── test_loader.py
├── .env.example               ← config template (copy to .env)
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

Pull the required models:
```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3             # answer generation
```

---

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd rag-project

# Create and activate virtual environment
python -m venv rag_venv
source rag_venv/bin/activate      # Windows: rag_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
```

---

## Usage

### Step 1 — Add your PDFs

Place PDF files inside `src/docs/` (gitignored due to size).

### Step 2 — Start Ollama

```bash
ollama serve
```

### Step 3 — Ingest documents (first time only)

```bash
cd src
../rag_venv/bin/python ingest.py

# To wipe and rebuild from scratch:
../rag_venv/bin/python ingest.py --rebuild
```

### Step 4a — Terminal chat

```bash
../rag_venv/bin/python coding.py
```

```
Loading vector store...
RAG ready (model=llama3, k=3). Type 'quit' to exit.

You: What is deep learning?
Answer: Deep learning is a subfield of machine learning...
Sources: Deep Learning by Ian Goodfellow.pdf | chunks: 3 | score: 0.312 | retrieval: 45ms | generation: 3200ms

Session: 1 queries | avg latency 3245ms | low-confidence answers: 0
```

### Step 4b — Streamlit web UI

```bash
../rag_venv/bin/streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 4c — Docker

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). Ollama must be running on your host.

---

## Configuration

All settings are in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3` | Ollama model for generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `RETRIEVAL_K` | `3` | Chunks retrieved per query |
| `RELEVANCE_THRESHOLD` | `0.65` | Max cosine distance before blocking |
| `CHUNK_SIZE` | `750` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

---

## Guardrails

| Guard | Behaviour |
|---|---|
| Short query (< 5 chars) | Blocked with message |
| Long query (> 500 chars) | Blocked with message |
| Prompt injection | Blocked (regex patterns) |
| Off-topic query | Allowed with a warning |
| Low retrieval relevance | Blocked before LLM call |

---

## Monitoring

Every query is logged to `src/data/rag_queries.jsonl`:

```json
{
  "ts": "2026-05-16T10:23:01.123456+00:00",
  "query": "What is an LSTM?",
  "answer": "An LSTM (Long Short-Term Memory)...",
  "sources": ["Deep Learning by Ian Goodfellow.pdf"],
  "retrieval_ms": 48.2,
  "generation_ms": 3105.7,
  "avg_chunk_score": 0.287
}
```

---

## Tests

```bash
rag_venv/bin/python -m pytest tests/ -v
```

31 tests covering guardrails, monitor, and document loader. CI runs automatically on every push via GitHub Actions.

---

## Books Used (not included in repo)

- *AI Engineering*
- *Applied Machine Learning and AI for Engineers*
- *Artificial Intelligence: A Modern Approach* — Russell & Norvig
- *Deep Learning* — Goodfellow, Bengio, Courville
- *GANs in Action*
