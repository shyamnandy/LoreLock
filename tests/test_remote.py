"""The Atlas round trip.

Hermetic: a dict-backed fake stands in for the driver, so the suite never
needs a MONGODB_URI, a network, or a running mongod. What is under test is
our serialisation and the hydrate path, not pymongo.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from app.store import db, remote, vectors


class _FakeCollection:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    def replace_one(self, flt, doc, upsert=False):
        self.docs[flt["_id"]] = copy.deepcopy(doc)

    def find_one(self, flt, *args, **kwargs):
        return copy.deepcopy(self.docs.get(flt["_id"]))

    def find(self, flt, projection=None):
        return [{"_id": key} for key in self.docs]

    def delete_one(self, flt):
        self.docs.pop(flt["_id"], None)


class _FakeDB:
    def __init__(self):
        self.books = _FakeCollection()
        self.payloads = _FakeCollection()


@pytest.fixture
def atlas(monkeypatch):
    """A fake Atlas that `remote` will talk to, with the module reporting
    itself enabled."""
    fake = _FakeDB()
    monkeypatch.setenv("MONGODB_URI", "mongodb://fake")
    monkeypatch.setattr(remote, "_database", lambda: fake)
    return fake


@pytest.fixture
def restored(fixture_book, conn):
    """Snapshot and restore the shared fixture book.

    `fixture_book` is session-scoped, so a test that simulates losing the
    disk removes rows every *later* test in the session still depends on --
    and pytest will not rebuild it, because the fixture is already cached.
    Tests below that wipe the index must take this fixture, not fixture_book.
    """
    book, chunks = fixture_book
    matrix = vectors.load(book.id).copy()
    yield book, chunks, matrix

    db.upsert_book(conn, book, "deadbeef")
    db.set_edition_pages(conn, book.id, book.edition_pages)
    db.insert_chunks(conn, chunks)
    vectors.save(book.id, matrix)
    vectors.load.cache_clear()


def test_disabled_without_uri(monkeypatch, conn):
    """No URI must be a silent no-op, not an error: local development and CI
    run this way."""
    monkeypatch.delenv("MONGODB_URI", raising=False)
    assert remote.enabled() is False
    assert remote.hydrate(conn) == 0
    assert remote.remote_book_ids() == []


def test_pack_round_trip_is_lossless(fixture_book, conn):
    book, chunks = fixture_book
    matrix = vectors.load(book.id)

    packed_chunks = remote._pack_chunks(chunks)
    packed_matrix = remote._pack_matrix(matrix)

    assert remote._unpack_chunks(packed_chunks) == chunks
    assert np.array_equal(remote._unpack_matrix(packed_matrix), matrix)


def test_oversized_book_is_refused(fixture_book):
    """A document over the Mongo cap must fail loudly at push time, not
    produce a book that silently never syncs."""
    with pytest.raises(ValueError, match="too long for one-document"):
        remote._guard_size("huge", b"x" * remote._SAFE_LIMIT, b"y" * 1024)


def test_push_then_hydrate_restores_the_book(atlas, restored, conn):
    book, chunks, matrix = restored

    remote.push_book(book, "deadbeef", chunks, matrix)
    assert remote.remote_book_ids() == [book.id]

    # lose the container's disk
    vectors.delete(book.id)
    db.delete_book(conn, book.id)
    vectors.load.cache_clear()
    assert db.get_book(conn, book.id) is None

    assert remote.hydrate(conn) == 1

    assert db.get_book(conn, book.id) == book
    assert db.all_chunks(conn, book.id) == chunks
    assert np.array_equal(vectors.load(book.id), matrix)


def test_hydrate_restores_the_readers_edition(atlas, restored, conn):
    """REGRESSION: `upsert_book` does not write `edition_pages` -- it belongs
    to `set_edition_pages` -- so hydrating through upsert alone silently reset
    a reader's declared edition to our own paging on every redeploy, moving
    every page number they had.
    """
    book, chunks, _ = restored
    db.set_edition_pages(conn, book.id, 321)
    declared = db.get_book(conn, book.id)
    assert declared.edition_pages == 321

    remote.push_book(declared, "deadbeef", chunks, vectors.load(book.id))

    db.set_edition_pages(conn, book.id, None)
    assert db.get_book(conn, book.id).edition_pages is None

    remote.hydrate(conn, force=True)

    assert db.get_book(conn, book.id).edition_pages == 321


def test_delete_is_mirrored(atlas, fixture_book, conn):
    """A book deleted locally but left in Atlas would reappear at the next
    boot, which reads as a failed delete."""
    book, chunks = fixture_book
    remote.push_book(book, "deadbeef", chunks, vectors.load(book.id))
    assert remote.remote_book_ids() == [book.id]

    remote.delete_book(book.id)
    assert remote.remote_book_ids() == []


def test_partial_write_is_skipped_not_trusted(atlas, restored, conn):
    """A chunk/matrix length mismatch means a torn write. Hydrating it would
    give a book whose prefix bound does not match its matrix -- the exact
    condition `vectors.prefix` raises on."""
    book, chunks, _ = restored
    remote.push_book(book, "deadbeef", chunks, vectors.load(book.id))

    # truncate the matrix behind the chunk list
    payload = atlas.payloads.docs[book.id]
    payload["matrix"] = remote._pack_matrix(vectors.load(book.id)[:-5])

    vectors.delete(book.id)
    db.delete_book(conn, book.id)
    vectors.load.cache_clear()

    assert remote.hydrate(conn) == 0
    assert db.get_book(conn, book.id) is None
