"""Phase 02 gate.

The last test in this file is the one that matters: it deliberately removes
the keyword path's bound and proves the suite goes red. A firewall test that
still passes when you break the firewall is decorative.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.models import FirewallBreach
from app.retrieval.firewall import SafeRetriever, _fts_query
from app.store import db, vectors
from tests.conftest import BOOK_ID, SECRET


def _positions(book):
    return [0, 400, 1500, 3000, book.total_words // 2, book.total_words]


def test_vector_operand_is_literally_a_prefix(fixture_book):
    book, chunks = fixture_book
    full = vectors.load(BOOK_ID)
    n = len(chunks)
    for n_p in {0, 1, n // 2, n}:
        sliced = vectors.prefix(BOOK_ID, n_p)
        assert sliced.shape[0] == n_p
        assert np.array_equal(sliced, full[:n_p])


def test_prefix_rejects_a_boundary_past_the_matrix(fixture_book):
    """Drift between the chunk table and the embeddings must fail loudly.

    Slicing past the end would silently clamp and hand back the whole book.
    """
    _, chunks = fixture_book
    with pytest.raises(ValueError, match="drifted"):
        vectors.prefix(BOOK_ID, len(chunks) + 1)


def test_no_retrieved_chunk_is_past_the_position(conn, fixture_book):
    book, _ = fixture_book
    for pos in _positions(book):
        r = SafeRetriever(conn, BOOK_ID, pos)
        for e in r.search("who is the culprit and what happened", k=8):
            assert e.chunk.word_end <= pos


@pytest.mark.parametrize("pos", [0, 250, 1200, 4000, 7000])
def test_secret_word_never_surfaces_before_its_position(conn, fixture_book, pos):
    """The reveal lives in the last chapter. Ask for it from earlier on."""
    book, chunks = fixture_book
    reveal_end = next(c.word_end for c in chunks if SECRET in c.text)
    r = SafeRetriever(conn, BOOK_ID, pos)
    texts = " ".join(e.chunk.text for e in r.search(f"{SECRET} culprit", k=8))
    if pos < reveal_end:
        assert SECRET not in texts


def test_shingle_leak_across_the_whole_book(conn, fixture_book):
    """No 8-word shingle from beyond the position may appear in the payload.

    Catches leaks that a per-chunk position check would miss -- overlap
    bleed, summaries, citation text.
    """
    book, chunks = fixture_book
    pos = 3000
    future_text = " ".join(c.text for c in chunks if c.word_start >= pos)
    fw = future_text.split()
    future_shingles = {" ".join(fw[i:i + 8]) for i in range(0, len(fw) - 8, 8)}

    r = SafeRetriever(conn, BOOK_ID, pos)
    payload = " ".join(e.chunk.text for e in r.search("culprit chapter events", k=8))
    pw = payload.split()
    payload_shingles = {" ".join(pw[i:i + 8]) for i in range(len(pw) - 8)}

    assert not (payload_shingles & future_shingles)


def test_position_zero_returns_nothing(conn, fixture_book):
    r = SafeRetriever(conn, BOOK_ID, 0)
    assert r.n_p == 0
    assert r.search("anything at all") == []


def test_advancing_position_only_widens_the_eligible_set(conn, fixture_book):
    book, _ = fixture_book
    seen: set[str] = set()
    for pos in sorted(_positions(book)):
        r = SafeRetriever(conn, BOOK_ID, pos)
        ids = {c.id for c in db.all_chunks(conn, BOOK_ID)[:r.n_p]}
        assert seen <= ids, "advancing the reader must never remove evidence"
        seen = ids


def test_assert_safe_raises_rather_than_filters(conn, fixture_book):
    book, chunks = fixture_book
    r = SafeRetriever(conn, BOOK_ID, 500)
    with pytest.raises(FirewallBreach):
        r.assert_safe([chunks[-1]])


def test_fts_query_never_passes_raw_user_text(conn):
    """User text is tokenised and re-quoted, so FTS5 operators cannot be
    injected to widen a query past its bound."""
    hostile = 'holmes" OR chunks_fts MATCH "zarquon'
    built = _fts_query(hostile)

    # every surviving token is individually double-quoted
    tokens = built.split(" OR ")
    assert tokens, "query must not be empty for word-bearing input"
    for tok in tokens:
        assert tok.startswith('"') and tok.endswith('"')
        assert '"' not in tok[1:-1], "no token may break out of its quotes"

    # and it is valid FTS5 syntax
    conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
                 (built,)).fetchall()


# -- the mutation test ------------------------------------------------------

def test_MUTATION_removing_the_keyword_bound_leaks(conn, fixture_book):
    """Break the firewall on purpose; prove the suite notices.

    This runs the keyword path with `c.idx < :n_p` removed. It must surface
    chunks from beyond the reader position, and `assert_safe` must catch
    them. If this test ever passes silently, the bound has stopped being
    load-bearing and the other tests in this file prove nothing.
    """
    book, chunks = fixture_book
    pos = 1000
    r = SafeRetriever(conn, BOOK_ID, pos)
    match = _fts_query(f"{SECRET} culprit")

    unbounded = conn.execute(
        """SELECT c.id AS id FROM chunks_fts f
             JOIN chunks c ON c.rowid = f.rowid
            WHERE chunks_fts MATCH ? AND c.book_id = ?
         ORDER BY bm25(chunks_fts) ASC LIMIT 8""",
        (match, BOOK_ID),
    ).fetchall()
    leaked = list(db.chunks_by_ids(conn, [x["id"] for x in unbounded]).values())

    assert any(c.word_end > pos for c in leaked), \
        "mutation produced no leak -- the fixture no longer exercises the bound"

    with pytest.raises(FirewallBreach):
        r.assert_safe(leaked)
