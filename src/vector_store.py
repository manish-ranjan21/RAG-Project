import time
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import config


class VectorStore:
    def __init__(self, model: str = config.EMBED_MODEL, db_path: str = config.DB_PATH):
        self.embeddings = HuggingFaceEmbeddings(model_name=model)
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore = None

    def create(self, chunks: List[Document], batch_size: int = 500):
        print(f"Creating embeddings for {len(chunks)} chunks...")
        if not chunks:
            raise ValueError("No chunks provided.")

        start = time.time()
        self.vectorstore = None
        num_batches = -(-len(chunks) // batch_size)

        with tqdm(total=num_batches, desc="Embedding batches") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i: i + batch_size]
                if self.vectorstore is None:
                    self.vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        persist_directory=str(self.db_path),
                        collection_name="rag_docs",
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                else:
                    self.vectorstore.add_documents(batch)
                pbar.update(1)

        elapsed = round(time.time() - start, 2)
        print(f"Vector Store created in {elapsed}s with {len(chunks)} chunks.")
        return self.vectorstore

    def create_in_memory(self, chunks: List[Document]):
        """Build a temporary in-memory vector store (used by Streamlit Cloud)."""
        if not chunks:
            raise ValueError("No chunks provided.")

        print(f"Building in-memory index for {len(chunks)} chunks...")
        tmp = tempfile.mkdtemp()
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=tmp,
            collection_name="rag_docs",
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"In-memory index ready with {len(chunks)} chunks.")
        return self.vectorstore

    def load(self):
        if not any(self.db_path.iterdir()):
            raise RuntimeError(
                f"No vector store found at {self.db_path}. Run ingest.py first."
            )
        self.vectorstore = Chroma(
            persist_directory=str(self.db_path),
            embedding_function=self.embeddings
        )
        print(f"Loaded vector store from {self.db_path}")
        return self.vectorstore

    def search(self, query: str, k: int = config.RETRIEVAL_K, with_scores: bool = False):
        if self.vectorstore is None:
            self.vectorstore = self.load()
        if with_scores:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        return self.vectorstore.similarity_search(query, k=k)

    def get_stats(self) -> dict:
        if self.vectorstore is None:
            self.vectorstore = self.load()
        try:
            count = self.vectorstore._collection.count()
            return {
                'total_chunks': count,
                'db_path': str(self.db_path),
                'embedding_model': self.embeddings.model_name
            }
        except Exception as e:
            return {'error': str(e)}

    def delete(self):
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore = None
            print("Vector Store deleted.")
