from pathlib import Path

from fastapi.testclient import TestClient

from rag.embeddings import HashingEmbedder
from rag.pipeline import RAGService

ROOT = Path(__file__).resolve().parent.parent


def test_api(tmp_path):
    import app.main as main
    main._service = RAGService(data_dir=str(tmp_path), embedder=HashingEmbedder(),
                               chunk_size=60, chunk_overlap=15)
    client = TestClient(main.app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/documents", files={"file": ("x.exe", b"nope")}).status_code == 400

    body = (ROOT / "sample_docs" / "faiss_vector_search.md").read_bytes()
    r = client.post("/documents", files={"file": ("faiss.md", body)})
    assert r.status_code == 200 and r.json()["chunks_indexed"] >= 1
    assert "faiss.md" in client.get("/documents").json()

    q = client.post("/query", json={"question": "What is product quantization?", "top_k": 2})
    assert q.status_code == 200 and q.json()["sources"][0]["doc"] == "faiss.md"
    assert client.post("/query", json={"question": ""}).status_code == 422

    assert client.delete("/documents/faiss.md").status_code == 200
    assert client.delete("/documents/faiss.md").status_code == 404