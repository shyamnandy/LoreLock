"""Synthetic fixture book.

Tests must not depend on a network fetch or a 5-minute embedding run, so the
fixture book is generated with deterministic pseudo-text and deterministic
pseudo-embeddings. The firewall is a property of ordering and slicing; it
does not care whether the vectors came from a real model.
"""
from __future__ import annotations

import os

# The suite must be hermetic: no network, no API key, no quota. `.env` is
# loaded by app.config at import time, so a setdefault here would lose to a
# real LLM_PROVIDER and the tests would silently start billing a provider --
# which is exactly what happened, and took the run from 10s to 191s.
os.environ["LLM_PROVIDER"] = "stub"
os.environ["VALIDATOR_USE_MODEL"] = "0"

import numpy as np                                          # noqa: E402
import pytest                                               # noqa: E402

from app.config import EMBED_DIM
from app.ingestion.chunker import Section, chunk_book, total_words
from app.models import Book
from app.store import db, vectors

BOOK_ID = "testbook"
# a marker word that only appears late in the book, used by the leak tests
SECRET = "zarquon"


def _make_sections(n_chapters: int = 8, words_per_chapter: int = 900) -> list[Section]:
    sections = []
    for ch in range(1, n_chapters + 1):
        words = [f"ch{ch}w{i}" for i in range(words_per_chapter)]
        # the reveal lives in the final chapter only
        if ch == n_chapters:
            words[words_per_chapter // 2] = SECRET
            words[words_per_chapter // 2 + 1] = "culprit"
        sections.append(Section(number=ch, title=f"Chapter {ch}",
                                text=" ".join(words)))
    return sections


@pytest.fixture(scope="session")
def fixture_book(tmp_path_factory):
    sections = _make_sections()
    chunks = chunk_book(BOOK_ID, sections)

    book = Book(id=BOOK_ID, title="Test Book", author="Fixture",
                source="synthetic", source_url="", license="test",
                total_words=total_words(sections), n_chunks=len(chunks))

    conn = db.connect()
    db.upsert_book(conn, book, checksum="deadbeef")
    db.insert_chunks(conn, chunks)

    # deterministic unit vectors, one per chunk
    rng = np.random.default_rng(1234)
    matrix = rng.standard_normal((len(chunks), EMBED_DIM)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    vectors.save(BOOK_ID, matrix)
    vectors.load.cache_clear()

    yield book, chunks

    conn.execute("DELETE FROM books WHERE id = ?", (BOOK_ID,))
    conn.commit()


@pytest.fixture
def conn():
    return db.connect()
