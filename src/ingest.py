from loader import DocumentLoader
from vector_store import VectorStore


def ingest(rebuild: bool = False):
    vs = VectorStore()

    if rebuild:
        print("Deleting existing vector store...")
        vs.delete()

    loader = DocumentLoader()
    docs = loader.load_pdf()

    if not docs:
        print("No PDFs found. Add PDF files to the docs/ folder and try again.")
        return

    chunks = loader.chunk_documents(docs)
    vs.create(chunks)
    stats = vs.get_stats()

    print("\nIngestion complete.")
    print(f"    Total chunks : {stats['total_chunks']}")
    print(f"    Database     : {stats['db_path']}")
    print(f"    Embed model  : {stats['embedding_model']}")


if __name__ == "__main__":
    import sys

    rebuild = "--rebuild" in sys.argv
    ingest(rebuild=rebuild)
