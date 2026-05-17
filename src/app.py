import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from loader import DocumentLoader
from vector_store import VectorStore
from monitor import Monitor
from guardrails import Guardrails
from rag_chain import RAGChain
import config

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")

# ── Read Groq API key (Streamlit secrets → .env fallback) ────
groq_api_key = st.secrets.get("GROQ_API_KEY", config.GROQ_API_KEY)

# ── Block app if API key is missing ──────────────────────────
if not groq_api_key:
    st.error("GROQ_API_KEY is not set.")
    st.markdown("""
### Setup Instructions

This app requires a free **Groq API key** to generate answers.

#### On Streamlit Cloud:
1. Open your app dashboard → **Settings → Secrets**
2. Add the following:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
3. Click **Save** — the app will restart automatically.

#### Running locally:
1. Copy `.env.example` to `.env`
2. Set your key:
```
GROQ_API_KEY=your_groq_api_key_here
```

#### Get a free Groq API key:
[console.groq.com](https://console.groq.com) → Sign up → Create API Key (free, no credit card needed)
""")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Chat")
    st.caption("Ask questions about your uploaded PDFs")
    st.divider()

    # PDF upload
    st.subheader("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Build index button
    st.subheader("2. Build Index")
    if st.button("Build Index", disabled=not uploaded_files or not groq_api_key, use_container_width=True):
        with st.spinner("Processing PDFs and building index..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    (Path(tmpdir) / f.name).write_bytes(f.getvalue())

                loader = DocumentLoader(folder=tmpdir)
                docs = loader.load_pdf()
                chunks = loader.chunk_documents(docs)

            vs = VectorStore()
            vs.create_in_memory(chunks)

            st.session_state.vs = vs
            st.session_state.ready = True
            st.session_state.messages = []
            st.session_state.monitor = Monitor()
            st.session_state.doc_names = [f.name for f in uploaded_files]

        st.success(f"Index ready — {len(chunks)} chunks from {len(uploaded_files)} PDF(s)")

    # Show loaded docs
    if st.session_state.get("doc_names"):
        st.caption("Loaded: " + ", ".join(st.session_state.doc_names))

    st.divider()

    # Settings
    st.subheader("Settings")
    model = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)
    k = st.slider("Chunks retrieved (k)", min_value=1, max_value=8, value=config.RETRIEVAL_K)

    st.divider()

    # Session stats
    st.subheader("Session Stats")
    stats_placeholder = st.empty()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Session state defaults ────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "monitor" not in st.session_state:
    st.session_state.monitor = Monitor()
if "ready" not in st.session_state:
    st.session_state.ready = False


# ── Main area ─────────────────────────────────────────────────
st.header("Chat with your PDFs")

if not st.session_state.ready:
    st.info("Upload PDFs and click **Build Index** in the sidebar to get started.")
    st.stop()

# Rebuild RAGChain each turn (model/k may change via sidebar)
rag = RAGChain(
    vector_store=st.session_state.vs,
    model=model,
    k=k,
    monitor=st.session_state.monitor,
    guardrails=Guardrails(),
    groq_api_key=groq_api_key
)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            if meta.get("topic_warning"):
                st.warning(meta["topic_warning"])
            if not meta.get("blocked"):
                cols = st.columns(4)
                cols[0].metric("Chunks", meta["num_chunks"])
                cols[1].metric("Relevance", meta["avg_chunk_score"])
                cols[2].metric("Retrieval", f"{meta['retrieval_ms']}ms")
                cols[3].metric("Generation", f"{meta['generation_ms']}ms")
                if meta["sources"]:
                    st.caption(f"Sources: {', '.join(meta['sources'])}")

# Input
query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.ask(query)

        if result.get("topic_warning"):
            st.warning(result["topic_warning"])

        st.markdown(result["answer"])

        if not result["blocked"]:
            cols = st.columns(4)
            cols[0].metric("Chunks", result["num_chunks"])
            cols[1].metric("Relevance", result["avg_chunk_score"])
            cols[2].metric("Retrieval", f"{result['retrieval_ms']}ms")
            cols[3].metric("Generation", f"{result['generation_ms']}ms")
            if result["sources"]:
                st.caption(f"Sources: {', '.join(result['sources'])}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": result
    })

    # Refresh sidebar stats
    stats = st.session_state.monitor.session_stats()
    with stats_placeholder:
        st.metric("Total queries", stats["total_queries"])
        st.metric("Avg latency", f"{stats['avg_latency_ms']}ms")
        st.metric("Low-confidence", stats["low_confidence_answers"])

    st.rerun()
