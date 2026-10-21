"""FastAPI service exposing document upload, listing, deletion and querying."""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.pipeline import RAGService

app = FastAPI(title="RAG Document Search", version="1.0.0")
_service: RAGService | None = None


def get_service() -> RAGService:
    """Return the shared RAGService, creating it on first use.

    Creation is deferred so that importing this module does not load the
    embedding model.
    """
    global _service
    if _service is None:
        _service = RAGService()
    return _service


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    """List indexed documents with their chunk counts."""
    return get_service().store.documents()


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Index an uploaded .pdf, .txt or .md file, replacing any previous version."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(400, "Only .pdf, .txt, and .md files are supported")
    # The loader reads from a path, so the upload is written to a temporary file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        n = get_service().ingest(tmp_path, name=Path(file.filename).name)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"document": file.filename, "chunks_indexed": n}


@app.delete("/documents/{name}")
def delete_document(name: str):
    if get_service().delete(name) == 0:
        raise HTTPException(404, f"No document named {name}")
    return {"deleted": name}


@app.post("/query")
def query(req: QueryRequest):
    """Answer a question, returning the answer and the ranked source passages."""
    return get_service().ask(req.question, req.top_k)