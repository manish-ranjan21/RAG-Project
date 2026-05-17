from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


class DocumentLoader:
    def __init__(self, folder: str = config.DOCS_FOLDER):
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True)

    def load_pdf(self) -> List[Document]:
        docs = []
        pdf_files = list(self.folder.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDFs.")

        for pdf in pdf_files:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            for page in pages:
                page.metadata['source_file'] = pdf.name
                page.metadata['total_pages'] = len(pages)
            docs.extend(pages)

        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = splitter.split_documents(docs)
        chunks = [
            c for c in chunks
            if len(c.page_content.strip()) > 50
            and c.page_content.count("\n") < 10
            and not c.page_content.strip().isdigit()
        ]

        print(f"Created {len(chunks)} clean chunks")
        return chunks
