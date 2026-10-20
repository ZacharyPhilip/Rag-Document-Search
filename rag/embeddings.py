"""Embedding models: turn text into lists of numbers (vectors).

Texts with similar meaning get vectors that point in similar directions, which
is what lets us search by meaning instead of by exact words.
"""
import os   
import re   
import zlib  

import numpy as np   


class SentenceTransformerEmbedder:
    """The real embedder: a small neural network that captures meaning."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)   # downloads the model on first use
        # dim = how many numbers are in each vector. The vector store needs to know this.
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


class HashingEmbedder:
    #A simple stand-in embedder that needs no downloads.

    def __init__(self, dim: int = 2048):
        self.dim = dim   

    def encode(self, texts: list[str]) -> np.ndarray:
        # One row per text, one column per slot, all starting at zero.
        out = np.zeros((len(texts), self.dim), dtype="float32")

        for row, text in enumerate(texts):
            # Pull out lowercase words and numbers, ignoring punctuation.
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if len(token) > 2:   # skip tiny words that dont matter "a", "to", "of"
                    out[row, zlib.crc32(token.encode()) % self.dim] += 1.0

        # log1p(x) = log(1 + x). It softens big counts so a word repeated 10 times
        # doesn't completely drown out every other word.
        out = np.log1p(out)

        norms = np.linalg.norm(out, axis=1, keepdims=True)   # length of rows
        norms[norms == 0] = 1.0  
        return (out / norms).astype("float32")


def get_embedder():
    """Pick an embedder. Setting the environment variable EMBEDDER=hash
    selects the lightweight one which is very fast otherwise we use the real model."""
    if os.getenv("EMBEDDER", "").lower() == "hash":
        return HashingEmbedder()
    # EMBED_MODEL lets you swap in a different model without editing code.
    return SentenceTransformerEmbedder(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))