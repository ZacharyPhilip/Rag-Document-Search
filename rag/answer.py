"""Extractive answering: build a response from retrieved passages

The sentences that share the most meaningful words with the question are
returned verbatim, each tagged with the number of the source it came from, so
every statement in an answer is cited and can be looked up in the document uploaded.
"""
import re

from .ingest import Chunk

#smaller words that don't matter to actual meaning of sentence as much
STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "our", "out", "how", "why", "what", "when", "where", "which", "who",
    "does", "did", "with", "this", "that", "from", "have", "has", "had", "its",
    "about", "into", "than", "then", "there", "these", "those", "they", "will",
}


def _words(text: str) -> set[str]:
    """Lowercased content words: drops stopwords and tokens of two characters or fewer."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 2 and w not in STOPWORDS}


def generate_answer(question: str, hits: list[tuple[Chunk, float]], max_sentences: int = 2) -> str:
    """Answer ``question`` from retrieved ``hits`` (chunk, score), best match first.

    Citations like ``[1]`` refer to the 1-based position of the chunk in ``hits``.
    """
    if not hits:
        return "No documents have been indexed yet. Upload a document first."

    q_words = _words(question)
    candidates = []  # (overlap, -source_rank, source_rank, sentence)
    for rank, (chunk, _) in enumerate(hits, 1):
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            sentence = sentence.strip()
            if len(sentence.split()) < 4:  # skip headings and fragments
                continue
            # Ties on word overlap are broken in favour of the higher-ranked chunk.
            candidates.append((len(q_words & _words(sentence)), -rank, rank, sentence))

    if not candidates or max(c[0] for c in candidates) == 0:
        top, _ = hits[0]
        return (f"I couldn't find a close match, but the closest passage was from "
                f"{top.doc} (page {top.page}) [1]. See the sources below.")

    best = sorted(candidates, reverse=True)[:max_sentences]
    best.sort(key=lambda c: (c[2], c[3]))  # present in source order
    return " ".join(f"{sentence} [{rank}]" for _, _, rank, sentence in best)