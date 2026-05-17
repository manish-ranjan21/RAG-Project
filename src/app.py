import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from vector_store import VectorStore
from monitor import Monitor
from guardrails import Guardrails
from rag_chain import RAGChain
import config

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 RAG Chat")
    st.caption("Ask questions about your AI/ML books")
    st.divider()

    st.subheader("Settings")
    model = st.selectbox("LLM Model", ["llama3", "mistral", "gemma2"], index=0)
    k = st.slider("Chunks retrieved (k)", min_value=1, max_value=8, value=config.RETRIEVAL_K)
    st.divider()

    st.subheader("Session Stats")
    stats_placeholder = st.empty()

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── Session state ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "monitor" not in st.session_state:
    st.session_state.monitor = Monitor()


# ── Load pipeline (cached so it only runs once) ───────────────

@st.cache_resource(show_spinner="Loading vector store...")
def load_pipeline(model: str, k: int):
    vs = VectorStore()
    vs.load()
    guardrails = Guardrails()
    return vs, guardrails


vs, guardrails = load_pipeline(model, k)
monitor: Monitor = st.session_state.monitor
rag = RAGChain(vs, model=model, k=k, monitor=monitor, guardrails=guardrails)


# ── Chat history ──────────────────────────────────────────────

st.header("Chat with your books")

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
                cols[1].metric("Relevance score", meta["avg_chunk_score"])
                cols[2].metric("Retrieval", f"{meta['retrieval_ms']}ms")
                cols[3].metric("Generation", f"{meta['generation_ms']}ms")
                if meta["sources"]:
                    st.caption(f"Sources: {', '.join(meta['sources'])}")


# ── Input ─────────────────────────────────────────────────────

query = st.chat_input("Ask a question about AI/ML...")

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
            cols[1].metric("Relevance score", result["avg_chunk_score"])
            cols[2].metric("Retrieval", f"{result['retrieval_ms']}ms")
            cols[3].metric("Generation", f"{result['generation_ms']}ms")
            if result["sources"]:
                st.caption(f"Sources: {', '.join(result['sources'])}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": result
    })

    # Update session stats in sidebar
    stats = monitor.session_stats()
    with stats_placeholder:
        st.metric("Total queries", stats["total_queries"])
        st.metric("Avg latency", f"{stats['avg_latency_ms']}ms")
        st.metric("Low-confidence", stats["low_confidence_answers"])

    st.rerun()
