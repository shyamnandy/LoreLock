"""The golden reveal set, as a test rather than a report.

`prompt_leakage` is the hard gate: whatever the provider does, a nonce that
first appears after the reader's position must never reach the prompt. The
answer-quality metrics live in scripts/evaluate.py because they depend on
which model is configured; this file asserts only what must hold for every
provider, including the keyless stub.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from app.config import RAW
from app.graph import ask
from app.ingestion.pipeline import ingest_file
from app.retrieval.firewall import SafeRetriever
from app.store import db

GOLDEN_PATH = pathlib.Path(__file__).parent / "golden_novella.json"
NOVELLA = RAW / "the-glasswing-affair.txt"

pytestmark = pytest.mark.skipif(
    not (GOLDEN_PATH.exists() and NOVELLA.exists()),
    reason="run scripts/make_novella.py first",
)


@pytest.fixture(scope="module")
def novella():
    golden = json.loads(GOLDEN_PATH.read_text())
    book = ingest_file(NOVELLA, book_id=golden["book_id"], title=golden["title"],
                       author=golden["author"], source="synthetic",
                       license="generated fixture")
    return book, golden


def test_every_nonce_is_unique_in_the_source(novella):
    _book, golden = novella
    text = NOVELLA.read_text(encoding="utf-8").lower()
    for q in golden["questions"]:
        assert text.count(q["answer_contains"].lower()) == 1, (
            f"{q['answer_contains']!r} is not unique; the leakage metric "
            f"would measure nothing"
        )


def test_nonce_never_reaches_the_prompt_before_its_reveal(novella):
    book, golden = novella
    for q in golden["questions"]:
        before = max(0, q["reveal_word"] - 400)
        out = ask("gold-before", book.id, q["question"], position=before)
        assert q["answer_contains"].lower() not in out.get("prompt_text", "").lower(), (
            f"{q['answer_contains']!r} reached the prompt at word {before}, "
            f"but is not revealed until {q['reveal_word']}"
        )
        assert q["answer_contains"].lower() not in out["answer"].lower()


def test_nonce_is_retrievable_once_the_reader_passes_it(novella):
    """Position consistency: the same evidence must become available."""
    book, golden = novella
    conn = db.connect()
    for q in golden["questions"]:
        after = min(book.total_words, q["reveal_word"] + 600)
        r = SafeRetriever(conn, book.id, after)
        eligible = " ".join(c.text for c in db.all_chunks(conn, book.id)[:r.n_p])
        assert q["answer_contains"].lower() in eligible.lower(), (
            f"{q['answer_contains']!r} still not eligible at word {after}"
        )


def test_eligible_text_grows_monotonically(novella):
    book, golden = novella
    conn = db.connect()
    previous = 0
    for pos in range(0, book.total_words, 2000):
        n_p = SafeRetriever(conn, book.id, pos).n_p
        assert n_p >= previous
        previous = n_p
