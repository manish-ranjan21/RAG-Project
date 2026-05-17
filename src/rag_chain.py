import time
from langchain_groq import ChatGroq
from monitor import Monitor
from guardrails import Guardrails
from vector_store import VectorStore
import config


class RAGChain:
    def __init__(self, vector_store: VectorStore, model: str = config.LLM_MODEL,
                 k: int = config.RETRIEVAL_K, monitor: Monitor = None,
                 guardrails: Guardrails = None, groq_api_key: str = None):
        self.vs = vector_store
        self.k = k
        self.llm = ChatGroq(
            model=model,
            temperature=0,
            api_key=groq_api_key or config.GROQ_API_KEY
        )
        self.monitor = monitor or Monitor()
        self.guardrails = guardrails or Guardrails()

    def ask(self, query: str) -> dict:
        # Input guardrails
        valid, reason = self.guardrails.check_input(query)
        if not valid:
            return {"answer": reason, "sources": [], "num_chunks": 0, "blocked": True}

        on_topic, topic_warning = self.guardrails.check_topic(query)

        # Retrieval + timing
        t0 = time.time()
        results = self.vs.search(query, k=self.k, with_scores=True)
        retrieval_ms = (time.time() - t0) * 1000

        docs = [doc for doc, _ in results]
        scores = [score for _, score in results]
        avg_score = sum(scores) / len(scores) if scores else 1.0
        context = "\n\n".join(d.page_content for d in docs)
        sources = list({d.metadata.get("source_file", "unknown") for d in docs})

        # Relevance guardrail
        relevant, relevance_warning = self.guardrails.check_relevance(avg_score)
        if not relevant:
            self.monitor.log(query, relevance_warning, sources, retrieval_ms, 0, avg_score)
            return {"answer": relevance_warning, "sources": sources, "num_chunks": len(docs), "blocked": True}

        # Generation + timing
        prompt = (
            "You are a helpful assistant. Use ONLY the context below to answer.\n"
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        t1 = time.time()
        response = self.llm.invoke(prompt)
        generation_ms = (time.time() - t1) * 1000

        answer = response.content
        self.monitor.log(query, answer, sources, retrieval_ms, generation_ms, avg_score)

        return {
            "answer": answer,
            "sources": sources,
            "num_chunks": len(docs),
            "avg_chunk_score": round(avg_score, 3),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "topic_warning": topic_warning if not on_topic else None,
            "blocked": False,
        }
