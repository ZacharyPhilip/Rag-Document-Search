from rag.embeddings import HashingEmbedder
from rag.ingest import chunk_pages
from rag.store import VectorStore


def test_store_search_save_load_remove(tmp_path):
    emb = HashingEmbedder()
    chunks = chunk_pages("a.txt", [(1, "apples oranges bananas fruit basket")], 50, 10) + \
             chunk_pages("b.txt", [(1, "kernel scheduler process thread cpu")], 50, 10)
    store = VectorStore(emb.dim)
    store.add(chunks, emb.encode([c.text for c in chunks]))
    top, _ = store.search(emb.encode(["fruit basket"])[0], k=1)[0]
    assert top.doc == "a.txt"

    store.save(tmp_path)
    reloaded = VectorStore.load(tmp_path, emb.dim)
    assert reloaded.documents() == store.documents()

    assert reloaded.remove_document("a.txt") == 1
    assert list(reloaded.documents()) == ["b.txt"]