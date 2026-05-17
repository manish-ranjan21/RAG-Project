from monitor import Monitor
from guardrails import Guardrails
from vector_store import VectorStore
from rag_chain import RAGChain
import config


def chat(model: str = config.LLM_MODEL, k: int = config.RETRIEVAL_K):
    """Interactive chat loop"""
    print("Loading vector store...")
    vs = VectorStore()
    vs.load()
    monitor = Monitor()
    guardrails = Guardrails()
    rag = RAGChain(vs, model=model, k=k, monitor=monitor, guardrails=guardrails)

    print(f"RAG ready (model={model}, k={k}). Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            stats = monitor.session_stats()
            print(f"\nSession: {stats['total_queries']} queries | "
                  f"avg latency {stats['avg_latency_ms']}ms | "
                  f"low-confidence answers: {stats['low_confidence_answers']}")
            print("Bye!")
            break

        result = rag.ask(query)

        if result.get("topic_warning"):
            print(f"[Warning] {result['topic_warning']}")

        print(f"\nAnswer: {result['answer']}")

        if not result["blocked"]:
            print(f"Sources: {', '.join(result['sources'])} | "
                  f"chunks: {result['num_chunks']} | "
                  f"score: {result['avg_chunk_score']} | "
                  f"retrieval: {result['retrieval_ms']}ms | "
                  f"generation: {result['generation_ms']}ms\n")


def test_search():
    vs = VectorStore()
    vectorstore = vs.load()
    stats = vs.get_stats()

    print(f"\nVector Store Statistics:")
    print(f"    Total chunks: {stats['total_chunks']}")
    print(f"    Database: {stats['db_path']}")
    print(f"    Model: {stats['embedding_model']}")

    test_queries = [
        "What is AI?",
        "What is Machine Learning?",
        "What is deep learning?",
        "What is LSTM?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"\nQuery: {query}")
        print('='*60)
        results = vectorstore.similarity_search_with_score(query, k=3)
        for doc, score in results:
            print(f"Score: {score:.3f}")
            print(f"Source: {doc.metadata.get('page', 'N/A')}")
            print(f"Content: {doc.page_content[:150]}...")


if __name__ == "__main__":
    chat()
