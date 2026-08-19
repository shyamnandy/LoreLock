"""Graph-level behaviour: the three outcomes, and the prompt boundary.

These run against the `stub` provider, so they exercise routing, the
citation checker and the prompt payload with no API key and no network.
"""
from __future__ import annotations

import os

import pytest

os.environ["LLM_PROVIDER"] = "stub"   # never call a real provider in tests

from app.graph import ask  # noqa: E402
from tests.conftest import BOOK_ID, SECRET  # noqa: E402


def test_thin_evidence_at_position_zero_refuses(fixture_book):
    out = ask("t-zero", BOOK_ID, "who is the culprit?", position=0)
    assert out["outcome"] in {"refused_thin", "refused_spoiler"}
    assert out["evidence"] == []


def test_prompt_never_contains_future_text(fixture_book):
    book, chunks = fixture_book
    pos = 3000
    out = ask("t-prompt", BOOK_ID, "what happened in the story", position=pos)
    prompt = out.get("prompt_text", "")
    for c in chunks:
        if c.word_start >= pos:
            assert c.text[:120] not in prompt, f"future chunk {c.id} reached the prompt"
    assert SECRET not in prompt


def test_every_evidence_item_is_within_position(fixture_book):
    book, _ = fixture_book
    for pos in (800, 2500, 5000, book.total_words):
        out = ask("t-bound", BOOK_ID, "describe the events", position=pos)
        for e in out.get("evidence", []):
            assert e.chunk.word_end <= pos


def test_outcome_is_always_one_of_three(fixture_book):
    book, _ = fixture_book
    for pos in (0, 1200, book.total_words):
        out = ask("t-out", BOOK_ID, "who is the culprit?", position=pos)
        assert out["outcome"] in {"answered", "refused_spoiler", "refused_thin"}
        assert out["answer"].strip()


def test_answer_carries_only_supplied_citations(fixture_book):
    book, _ = fixture_book
    out = ask("t-cite", BOOK_ID, "describe the events", position=book.total_words)
    supplied = {e.chunk.id for e in out.get("evidence", [])}
    for cid in out.get("citations", []):
        assert cid in supplied


def test_conversation_log_records_the_actual_prompt(fixture_book):
    import hashlib

    from app.store import db
    out = ask("t-log", BOOK_ID, "what happened", position=4000)
    row = db.connect().execute(
        "SELECT prompt_hash, prompt_text FROM conversations "
        "WHERE session_id='t-log' ORDER BY id DESC LIMIT 1").fetchone()
    assert row["prompt_hash"] == hashlib.sha256(row["prompt_text"].encode()).hexdigest()
    assert row["prompt_text"] == out.get("prompt_text", "")
