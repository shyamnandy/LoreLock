"""Phase 01 gate: the ordering invariant the whole firewall rests on."""
from __future__ import annotations

import pytest

from app.ingestion.chunker import Section, chunk_book
from app.store import db


def test_idx_is_monotonic_in_both_offsets(fixture_book):
    _, chunks = fixture_book
    for a, b in zip(chunks, chunks[1:]):
        assert b.idx == a.idx + 1
        assert b.word_start >= a.word_start
        assert b.word_end >= a.word_end, (
            "word_end must be monotonic or the eligible set is not a prefix"
        )


def test_word_offsets_are_continuous_across_chapters():
    secs = [Section(1, "One", " ".join(f"a{i}" for i in range(400))),
            Section(2, "Two", " ".join(f"b{i}" for i in range(400)))]
    chunks = chunk_book("bk", secs)
    boundary = [c for c in chunks if c.chapter == 2][0]
    last_ch1 = [c for c in chunks if c.chapter == 1][-1]
    assert boundary.word_start >= last_ch1.word_start
    assert boundary.word_start == 400, "chapter 2 must start at the global offset"


def test_chunks_never_straddle_a_chapter(fixture_book):
    _, chunks = fixture_book
    assert all(isinstance(c.chapter, int) for c in chunks)
    for ch in {c.chapter for c in chunks}:
        idxs = [c.idx for c in chunks if c.chapter == ch]
        assert idxs == list(range(min(idxs), max(idxs) + 1)), \
            "a chapter's chunks must be contiguous in idx"


def test_insert_rejects_non_monotonic_chunks(fixture_book):
    _, chunks = fixture_book
    scrambled = [chunks[2], chunks[0], chunks[1]]
    with pytest.raises(ValueError):
        db.insert_chunks(db.connect(), scrambled)


def test_n_p_uses_word_end_not_word_start(conn, fixture_book):
    """A chunk that merely *starts* before the reader is not yet eligible."""
    _, chunks = fixture_book
    c = chunks[3]
    mid = (c.word_start + c.word_end) // 2
    n_p = db.n_p_for_position(conn, "testbook", mid)
    eligible = chunks[:n_p]
    assert all(e.word_end <= mid for e in eligible)
    assert c not in eligible, "partially-read chunk must be withheld"
