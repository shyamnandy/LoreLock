from __future__ import annotations

import pytest

from app import memory
from app.models import Evidence, FirewallBreach
from tests.conftest import BOOK_ID


def _ev(chunk):
    return Evidence(chunk=chunk, score=1.0, via="vector", similarity=0.9)


def test_memory_rejects_a_future_fact_on_write(conn, fixture_book):
    _, chunks = fixture_book
    late = chunks[-1]
    with pytest.raises(FirewallBreach):
        memory.remember(conn, "s1", "the culprit is revealed", [_ev(late)], 500)


def test_memory_accepts_and_recalls_a_past_fact(conn, fixture_book):
    _, chunks = fixture_book
    early = chunks[1]
    conn.execute("DELETE FROM memory_facts WHERE session_id = 's2'")
    conn.execute("INSERT OR IGNORE INTO books (id,title,author,source,source_url,"
                 "license,checksum,total_words,n_chunks) VALUES "
                 "(?,?,?,?,?,?,?,?,?)", (BOOK_ID, "t", "a", "s", "", "l", "c", 1, 1))
    conn.execute("INSERT OR REPLACE INTO sessions (id,book_id,position,n_p) "
                 "VALUES ('s2',?,?,?)", (BOOK_ID, 10_000, 0))
    memory.remember(conn, "s2", "a thing happened", [_ev(early)], early.word_end)

    assert len(memory.recall(conn, "s2", early.word_end)) == 1
    assert memory.recall(conn, "s2", early.word_end - 1) == []


def test_every_recalled_fact_is_within_position(conn, fixture_book):
    for f in memory.recall(conn, "s2", 999_999):
        assert f.max_position <= 999_999
