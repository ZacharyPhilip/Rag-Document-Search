from rag.answer import generate_answer
from rag.ingest import Chunk


def test_no_hits_message():
    assert "No documents" in generate_answer("anything", [])


def test_picks_best_sentence_and_cites():
    chunk = Chunk("notes.md", 1, 0,
                  "Cats sleep a lot during the day. The late policy removes ten percent per day. "
                  "Dogs enjoy long walks outside.")
    answer = generate_answer("What is the late policy?", [(chunk, 0.9)], max_sentences=1)
    assert "late policy" in answer and answer.endswith("[1]")


def test_no_overlap_falls_back_to_closest_passage():
    chunk = Chunk("notes.md", 2, 0, "Completely unrelated words about gardening and tomatoes here.")
    answer = generate_answer("zebra giraffe", [(chunk, 0.1)])
    assert "closest passage" in answer and "notes.md" in answer