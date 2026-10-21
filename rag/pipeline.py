"""High-level service that ties ingestion, embedding, storage and answering together."""
import os
from pathlib import Path

from .answer import generate_answer
from .embeddings import get_embedder
from .ingest import chunk_pages, load_pages
from .store import VectorStore


class RAGService:
    """Index documents and answer questions about them.

    Defaults can be overridden with the DATA_DIR, CHUNK_WORDS and CHUNK_OVERLAP
    environment variables, or by passing arguments.
    """

    def __init__(self, data_dir: str | None = None, embedder=None,
                 chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR", "data"))
        self.embedder = embedder or get_embedder()
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_WORDS", "200"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "40"))
        self.store = VectorStore.load(self.data_dir, self.embedder.dim)

    def ingest(self, path: str | Path, name: str | None = None) -> int:
        """Index a file and persist the store. Returns the number of chunks added.

        Re-ingesting a file with the same name replaces its previous chunks.
        """
        name = name or Path(path).name
        chunks = chunk_pages(name, load_pages(path), self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError("No extractable text found in document")
        self.store.remove_document(name)
        self.store.add(chunks, self.embedder.encode([c.text for c in chunks]))
        self.store.save(self.data_dir)
        return len(chunks)

    def delete(self, name: str) -> int:
        """Remove a document from the index. Returns that number of chunks removed."""
        removed = self.store.remove_document(name)
        if removed:
            self.store.save(self.data_dir)
        return removed

    def retrieve(self, question: str, k: int = 4):
        """Return the ``k`` chunks most similar to the question, with scores."""
        return self.store.search(self.embedder.encode([question])[0], k)

    def ask(self, question: str, k: int = 4) -> dict:
        """Answer a question and return the supporting sources alongside it."""
        hits = self.retrieve(question, k)
        return {
            "answer": generate_answer(question, hits),
            "sources": [
                {"doc": c.doc, "page": c.page, "score": round(s, 4), "text": c.text}
                for c, s in hits
            ],
        }