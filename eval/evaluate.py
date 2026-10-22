"""Measure retrieval quality with Hit@k and Mean Reciprocal Rank (MRR).

A retrieval counts as correct when a returned chunk comes from the expected
document and contains the expected keyword. Hit@k is the fraction of questions
with a correct chunk in the top k; MRR averages 1/rank of the first correct chunk.

Usage:
    python -m eval.evaluate --k 3 --chunk-words 60 --overlap 15 [--embedder hash]
"""
import argparse
import json
import os
import tempfile
from pathlib import Path


def evaluate(service, eval_set: list[dict], k: int) -> dict:
    """Score ``service`` on ``eval_set``, a list of {question, doc, keyword} items."""
    hits, reciprocal_ranks, misses = 0, 0.0, []
    for item in eval_set:
        results = service.retrieve(item["question"], k)
        rank = next(
            (r for r, (chunk, _) in enumerate(results, 1)
             if chunk.doc == item["doc"] and item["keyword"].lower() in chunk.text.lower()),
            None,
        )
        if rank:
            hits += 1
            reciprocal_ranks += 1 / rank
        else:
            misses.append(item["question"])
    n = len(eval_set)
    return {"n": n, "k": k, f"hit@{k}": hits / n, "mrr": reciprocal_ranks / n, "misses": misses}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--docs", default="sample_docs")
    ap.add_argument("--eval-set", default="eval/eval_set.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--chunk-words", type=int, default=60)
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--embedder", choices=["minilm", "hash"], default="minilm")
    args = ap.parse_args()

    if args.embedder == "hash":
        os.environ["EMBEDDER"] = "hash"
    # Imported after the environment variable is set, since embedder choice is read from it.
    from rag.pipeline import RAGService

    # A temporary index keeps evaluation runs from touching the real data directory.
    with tempfile.TemporaryDirectory() as tmp:
        service = RAGService(data_dir=tmp, chunk_size=args.chunk_words, chunk_overlap=args.overlap)
        for path in sorted(Path(args.docs).glob("*")):
            if path.suffix.lower() in {".md", ".txt", ".pdf"}:
                service.ingest(path)
        report = evaluate(service, json.loads(Path(args.eval_set).read_text()), args.k)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()