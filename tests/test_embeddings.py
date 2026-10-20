import numpy as np

from rag.embeddings import HashingEmbedder


def test_vectors_are_normalised():
    vecs = HashingEmbedder().encode(["hello world example", "another sentence here"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_similar_text_scores_higher():
    emb = HashingEmbedder()
    q, a, b = emb.encode([
        "python asyncio timeout",
        "asyncio timeout handling in python",
        "kernel scheduler thread",
    ])
    assert q @ a > q @ b