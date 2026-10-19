import pytest

from rag.ingest import chunk_pages, load_pages


def test_chunking_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_pages("d.txt", [(1, text)], size=30, overlap=10)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert "w99" in chunks[-1].text
    assert all(len(c.text.split()) <= 30 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_pages("d.txt", [(1, "a b c")], size=5, overlap=5)


def test_load_pages_reads_text_files(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello there world")
    assert load_pages(f) == [(1, "hello there world")]


def test_load_pages_rejects_unknown_types(tmp_path):
    f = tmp_path / "program.exe"
    f.write_text("nope")
    with pytest.raises(ValueError):
        load_pages(f)