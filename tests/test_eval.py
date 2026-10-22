import json
from pathlib import Path

from eval.evaluate import evaluate
from rag.embeddings import HashingEmbedder
from rag.pipeline import RAGService

ROOT = Path(__file__).resolve().parent.parent


def test_retrieval_quality_on_eval_set(tmp_path):
    svc = RAGService(data_dir=str(tmp_path), embedder=HashingEmbedder(),
                     chunk_size=60, chunk_overlap=15)
    for p in (ROOT / "sample_docs").glob("*.md"):
        svc.ingest(p)
    report = evaluate(svc, json.loads((ROOT / "eval" / "eval_set.json").read_text()), k=3)
    assert report["hit@3"] >= 0.7, report["misses"]