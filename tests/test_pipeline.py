from pathlib import Path

from rag.embeddings import HashingEmbedder
from rag.pipeline import RAGService

ROOT = Path(__file__).resolve().parent.parent


def test_pipeline_end_to_end_and_reupload_replaces(tmp_path):
    svc = RAGService(data_dir=str(tmp_path / "data"), embedder=HashingEmbedder(),
                     chunk_size=60, chunk_overlap=15)
    doc = ROOT / "sample_docs" / "python_asyncio.md"
    first = svc.ingest(doc)
    assert svc.ingest(doc) == first            # re-ingest replaces, no duplicates
    assert svc.store.documents() == {"python_asyncio.md": first}
    result = svc.ask("How do I add a timeout to an awaitable?")
    assert result["sources"][0]["doc"] == "python_asyncio.md"
    assert "wait_for" in result["answer"] and "[1]" in result["answer"]


def test_index_survives_restart(tmp_path):
    kwargs = dict(data_dir=str(tmp_path), embedder=HashingEmbedder(), chunk_size=60, chunk_overlap=15)
    RAGService(**kwargs).ingest(ROOT / "sample_docs" / "faiss_vector_search.md")
    reopened = RAGService(**kwargs)
    assert "faiss_vector_search.md" in reopened.store.documents()