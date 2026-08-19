"""Position-aware chunking.

Chunks are cut from the WHOLE book, once, before any reader position is
known. Boundaries must never move when the reader advances -- if they did,
the boundary chunk's text would change, its embedding would change, and
retrieval would stop being reproducible across positions.

Chunks are cut within chapters (never straddling a chapter break) but word
offsets are global and continuous, so `idx` stays monotonic in both
`word_start` and `word_end` across the whole book.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import CHUNK_OVERLAP, CHUNK_WORDS
from app.models import Chunk

_WS = re.compile(r"\s+")

# Split after . ! ? " ' ) closers, when followed by whitespace and something
# that looks like a new sentence. Deliberately conservative: a bad split is
# cheaper than a run-on, and abbreviations ("Mr.", "Mrs.") are guarded.
_SENTENCE = re.compile(
    r'(?:(?<=[.!?])|(?<=[.!?][")”’]))\s+(?=[A-Z“"‘\'])'
)
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Lt|Col|Capt|Rev|Hon|Jr|Sr|vs|etc)\.$")


def _sentences(text: str) -> list[str]:
    """Split into sentences, re-joining splits that landed after a title."""
    parts = _SENTENCE.split(text.strip())
    out: list[str] = []
    for part in parts:
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return [p for p in out if p.strip()]


@dataclass(frozen=True, slots=True)
class Section:
    """One chapter or titled division of the book, in reading order."""
    number: int
    title: str
    text: str


def _words(text: str) -> list[str]:
    return [w for w in _WS.split(text.strip()) if w]


def chunk_book(book_id: str, sections: list[Section]) -> list[Chunk]:
    """Cut `sections` into overlapping, position-tagged chunks."""
    stride = CHUNK_WORDS - CHUNK_OVERLAP
    if stride <= 0:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_WORDS")

    chunks: list[Chunk] = []
    cursor = 0   # global word offset, continuous across sections
    idx = 0

    for sec in sections:
        sentences = _sentences(sec.text)
        if not sentences:
            continue

        # word counts per sentence, and the running offset each one starts at
        lengths = [len(_words(s)) for s in sentences]
        starts: list[int] = []
        running = 0
        for n in lengths:
            starts.append(running)
            running += n
        section_words = running
        if section_words == 0:
            continue

        i = 0
        while i < len(sentences):
            # take whole sentences until the window is full, so a chunk never
            # begins or ends mid-sentence -- retrieved evidence that starts
            # "shocking! He deserves..." reads as noise to a model and to a
            # reader alike
            j, taken = i, 0
            while j < len(sentences) and (taken == 0 or taken + lengths[j] <= CHUNK_WORDS):
                taken += lengths[j]
                j += 1

            w_start = cursor + starts[i]
            w_end = w_start + taken
            chunks.append(Chunk(
                id=f"{book_id}::c{idx:05d}",
                book_id=book_id,
                idx=idx,
                text=" ".join(sentences[i:j]),
                chapter=sec.number,
                chapter_title=sec.title,
                word_start=w_start,
                word_end=w_end,
            ))
            idx += 1

            if j >= len(sentences):
                break

            # step back over whole sentences to build the overlap
            back, overlap = j, 0
            while back > i + 1 and overlap + lengths[back - 1] <= CHUNK_OVERLAP:
                back -= 1
                overlap += lengths[back]
            i = back if back > i else j

        cursor += section_words

    return chunks


def total_words(sections: list[Section]) -> int:
    return sum(len(_words(s.text)) for s in sections)
