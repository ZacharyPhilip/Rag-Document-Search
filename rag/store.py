"""In-memory vector store backed by FAISS, with on-disk persistence.

Chunks and their embeddings are kept in parallel: row i of ``vectors`` is the
embedding of ``chunks[i]``. Embeddings are expected to be L2-normalised, so
inner-product search is equivalent to cosine similarity.
"""
import json
from pathlib import Path
import faiss
import numpy as np
from .ingest import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, dim), dtype="float32")
        # Exact (brute-force) inner-product search; sufficient for thousands of chunks.
        self.index = faiss.IndexFlatIP(dim)

    def _rebuild(self) -> None:
        # IndexFlatIP cannot delete rows, so the index is rebuilt from self.vectors.
        self.index = faiss.IndexFlatIP(self.dim)
        if len(self.vectors):
            self.index.add(self.vectors)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length!")
        vectors = vectors.astype("float32")
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors])
        self.index.add(vectors)

    def remove_document(self, doc: str) -> int:
        """Remove all chunks from ``doc`` and returns that number removed."""
        keep = [i for i, c in enumerate(self.chunks) if c.doc != doc]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep]
            self._rebuild()
        return removed

    def documents(self) -> dict[str, int]:
        """Map each stored document name to its chunk count."""
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.doc] = counts.get(c.doc, 0) + 1
        return counts

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return up to ``k`` (chunk, similarity) pairs, best match first."""
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        # FAISS pads missing results with -1.
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def save(self, directory: str | Path) -> None:
        # The FAISS index is not saved; it is rebuilt from the vectors on load.
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks]), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, dim: int) -> "VectorStore":
        """Load a saved store, or return an empty one if none exists.

        Saved data with a different embedding dimension (e.g. after switching
        models) is ignored rather than raising.
        """
        directory = Path(directory)
        store = cls(dim)
        vec_path, chunk_path = directory / "vectors.npy", directory / "chunks.json"
        if vec_path.exists() and chunk_path.exists():
            vectors = np.load(vec_path)
            if vectors.ndim == 2 and vectors.shape[1] == dim:
                store.chunks = [Chunk(**d) for d in json.loads(chunk_path.read_text("utf-8"))]
                store.vectors = vectors.astype("float32")
                store._rebuild()
        return store