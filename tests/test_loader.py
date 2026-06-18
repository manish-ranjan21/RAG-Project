import pytest
from langchain_core.documents import Document

from loader import DocumentLoader


@pytest.fixture
def loader(tmp_path):
    return DocumentLoader(folder=str(tmp_path))


def make_doc(text: str, source: str = "test.pdf") -> Document:
    return Document(page_content=text, metadata={"source_file": source})


# ── chunk_documents ───────────────────────────────────────────


def test_chunks_normal_text(loader):
    docs = [make_doc("Machine learning is a subset of AI. " * 30)]
    chunks = loader.chunk_documents(docs)
    assert len(chunks) > 0


def test_filters_short_chunks(loader):
    docs = [make_doc("Hi")]  # less than 50 chars — should be filtered
    chunks = loader.chunk_documents(docs)
    assert len(chunks) == 0


def test_filters_pure_digit_chunks(loader):
    docs = [make_doc("123")]
    chunks = loader.chunk_documents(docs)
    assert len(chunks) == 0


def test_filters_header_chunks(loader):
    # 11 newlines — should be filtered (> 10 newlines)
    docs = [make_doc("\n" * 11 + "Some text here")]
    chunks = loader.chunk_documents(docs)
    assert len(chunks) == 0


def test_chunk_preserves_metadata(loader):
    docs = [make_doc("Deep learning uses neural networks with many layers. " * 20, source="dl.pdf")]
    chunks = loader.chunk_documents(docs)
    assert all(c.metadata.get("source_file") == "dl.pdf" for c in chunks)


def test_docs_folder_created(tmp_path):
    folder = tmp_path / "new_docs"
    DocumentLoader(folder=str(folder))
    assert folder.exists()
