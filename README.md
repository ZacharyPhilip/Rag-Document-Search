# RAG Document Search Project

Upload PDFs, text, or MD files and ask questions in English. The app finds the passages
most relevant to your question using semantic vector search, then returns the sentences that best
answer it, each cited with its source file and page. Everything runs locally, with no API keys and at no cost.

**Stack:** Python, FastAPI, FAISS, sentence-transformers (all-MiniLM-L6-v2), pypdf, pytest

## How it works

```
upload -> extract text -> split into overlapping word chunks -> embed chunks -> saved to disk with FAISS index

question -> embed -> top-k nearest chunks -> pick best-matching sentences -> answer with [1], [2] citations
```

1. **Ingest.** PDF, TXT, and MD files are read page by page and split into overlapping word windows,
   so each chunk keeps its page number for citations and answers that straddle a boundary are not lost.
2. **Embed and index.** Each chunk is embedded with `all-MiniLM-L6-v2`. Vectors are normalised, so
   an inner-product FAISS index (`IndexFlatIP`) gives cosine similarity.
3. **Retrieve.** The question is embedded and the `top_k` closest chunks are returned with scores.
4. **Answer.** The sentences in those chunks that share the most meaningful words with the question
   are returned verbatim, tagged with the number of the source they came from. Everything can be looked up in the offical document uploaded.

## Quick start

```
git clone https://github.com/ZacharyPhilip/rag-document-search.git
cd rag-document-search
python -m venv .venv
.venv\Scripts\activate          
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to test it out and be able to upload files and see the result!

The first upload downloads the embedding model. You may see a Hugging Face warning
about unauthenticated requests, it is harmless and doesn't affect the program.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/documents` | Upload and index a `.pdf`, `.txt`, or `.md` file (re-uploading a doc replaces it) |
| GET | `/documents` | List indexed documents with chunk counts |
| DELETE | `/documents/{name}` | Remove a document from the index |
| POST | `/query` | `{"question": "...", "top_k": 3}` returns an answer and ranked source passages |
| GET | `/health` | Health check |

Example response from `/query`:

```json
{
  "answer": "Late submissions of homework assignments will be accepted up to 24 hours after the due date [1]",
  "sources": [
    {"doc": "syllabus.pdf", "page": 6, "score": 0.354, "text": "..."}
  ]
}
```

## Configuration

Set these environment variables to override the defaults.

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_WORDS` | `100` | Words per chunk, increase if you want more of the passage to be read |
| `CHUNK_OVERLAP` | `20` | Words shared between neighbouring chunks |
| `DATA_DIR` | `data` | Where the index is saved *included in the gitignore file |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `EMBEDDER` | unset | Set to `hash` for a lightweight offline embedder (used in teststing suites) |

## Evaluation

Retrieval quality is measured with **Hit@k** (a correct chunk appears in the top k results) and
**MRR** (mean reciprocal rank of the first correct chunk). A result counts as correct when it comes
from the expected document and contains the expected keyword. The test set is in
`eval/eval_set.json`: 9 questions over the 3 documents in `sample_docs/`.

```
python -m eval.evaluate --k 3 --chunk-words 100 --overlap 20
```

Results with MiniLM:

| Chunk words / overlap | Hit@1 | Hit@3 | MRR (k=3) |
|---|---|---|---|
| 40 / 10 | 67% | 89% | 0.74 |
| 60 / 15 | 67% | 100% | 0.80 |
| 100 / 20 | 89% | 100% | 0.94 |

Larger chunks improved every metric on this set, so 100 / 20 is the default. One question
("How can vector memory use be reduced?") misses at every size: the document says vectors are
compressed to reduce memory use, and the question's wording does not overlap closely enough with
that sentence.

## Tests

```
pytest -q
```
The 14 tests cover chunking, embedding, the vector store (search, save, load, remove), answer
extraction, the pipeline, retrieval quality, and the API. They use the offline `hash` embedder, so
they run quickly with no model download.

## Project structure

```
app/main.py            FastAPI endpoints
rag/ingest.py          file loading and chunking
rag/embeddings.py      MiniLM and offline hashing embedders
rag/store.py           FAISS vector store with persistence
rag/answer.py          extractive answers with citations
rag/pipeline.py        RAGService tying the steps together
eval/evaluate.py       Hit@k and MRR evaluation
eval/eval_set.json     evaluation questions
sample_docs/           documents used by tests and evaluation
tests/                 pytest suite
```

## Limitations/Ideas for the future 

- **Answers are extracted, not written.** The app returns sentences from your documents. It does
  not paraphrase or combine information across passages, and when several sentences tie on word
  overlap it can include a less relevant one.
- **PDF text extraction is imperfect.** Headings are often merged into the body text and appear
  twice, and tables can come out jumbled. Scanned PDFs (images of text) have no extractable text,
  and there is no OCR.
-**small front end design