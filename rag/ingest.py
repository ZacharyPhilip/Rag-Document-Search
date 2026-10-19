"""Document loading and chunking.
Step one of the pipeline
"""
from dataclasses import dataclass, asdict
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    doc: str     
    page: int    
    index: int   
    text: str
    #Turn the chunk into a dictonary
    def to_dict(self) -> dict:
        return asdict(self)

#Read a file and return a list of (page_number, text) pairs
def load_pages(path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for a PDF, TXT, or MD file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    #only accepts PDF,TXT, and MD files. 
    raise ValueError(f"Unsupported file type: {suffix} (use .pdf, .txt, or .md)")

#Split the pages into overlapping section of words. 
def chunk_pages(doc: str, pages: list[tuple[int, str]],
                size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split pages into overlapping word windows.

    Chunking per page keeps page-number citations accurate. Overlap stops an
    answer that straddles a boundary from being cut in half.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks: list[Chunk] = []
    for page_num, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            window = words[start:start + size]
            chunks.append(Chunk(doc, page_num, len(chunks), " ".join(window)))
            if start + size >= len(words):
                break
    return chunks