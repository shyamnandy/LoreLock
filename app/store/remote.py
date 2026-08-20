"""Durable copy of the index in MongoDB Atlas.

Why this exists, and why it is NOT a vector database:

The retrieval path is deliberately file-backed. The firewall's strongest
bound is structural -- the operand of the similarity search *is* the shape
of `E[:n_p]`, so future rows are absent rather than filtered (see
`app.store.vectors`). Moving search to Atlas `$vectorSearch` would express
that bound as `filter: {idx: {$lt: n_p}}`, demoting the structural guarantee
to a predicate: the weaker of the two categories `app.retrieval.firewall`
distinguishes, and the one it keeps a mutation test against. FTS5 has no
Atlas equivalent either without a second rewrite.

So Atlas stores the *artifacts*, not the index. On boot a container with an
empty DATA_DIR rebuilds SQLite, the FTS index and the `.npy` matrices from
here. Retrieval never touches this module.

    Atlas    system of record   (survives the container)
    DATA_DIR working copy       (rebuilt on boot, may be ephemeral)

Everything degrades to a no-op when MONGODB_URI is unset, so local
development and the test suite are unaffected.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import zlib
from typing import TYPE_CHECKING

import numpy as np

from app.models import Book, Chunk

if TYPE_CHECKING:
    import sqlite3

# A single document holds one book's chunks and its embedding matrix. Mongo
# caps a document at 16 MB; a 600k-word novel is ~2,400 chunks, which is
# ~5 MB of text before compression and ~3.5 MB of float32. Comfortable, but
# checked rather than assumed -- see _guard_size.
_DOC_LIMIT = 16 * 1024 * 1024
_SAFE_LIMIT = 12 * 1024 * 1024

_client = None
_warned = False


def enabled() -> bool:
    return bool(os.environ.get("MONGODB_URI"))


def _database():
    """Cached client. serverSelectionTimeoutMS is short on purpose: a boot
    must not hang for 30s because Atlas is unreachable -- hydration is best
    effort, and the API still serves whatever is already on disk."""
    global _client
    if _client is None:
        from pymongo import MongoClient

        _client = MongoClient(
            os.environ["MONGODB_URI"],
            serverSelectionTimeoutMS=8_000,
            connectTimeoutMS=8_000,
            appname="reading-companion",
        )
    return _client[os.environ.get("MONGODB_DB", "reading_companion")]


# -- serialisation ---------------------------------------------------------

def _pack_chunks(chunks: list[Chunk]) -> bytes:
    """zlib'd JSON. Chunk text is highly compressible (~3x) and it is the
    half of the payload that grows with book length."""
    rows = [dataclasses.asdict(c) for c in chunks]
    return zlib.compress(json.dumps(rows).encode("utf-8"), level=6)


def _unpack_chunks(blob: bytes) -> list[Chunk]:
    return [Chunk(**r) for r in json.loads(zlib.decompress(blob).decode("utf-8"))]


def _pack_matrix(matrix: np.ndarray) -> bytes:
    """Raw .npy bytes. float32 embeddings are near-incompressible, so zlib
    would spend CPU for ~5% -- store them as-is and keep the round trip cheap."""
    buf = io.BytesIO()
    np.save(buf, matrix.astype(np.float32))
    return buf.getvalue()


def _unpack_matrix(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


def _guard_size(book_id: str, chunk_blob: bytes, matrix_blob: bytes) -> None:
    total = len(chunk_blob) + len(matrix_blob)
    if total > _SAFE_LIMIT:
        raise ValueError(
            f"{book_id!r} packs to {total / 1e6:.1f} MB, over the "
            f"{_SAFE_LIMIT / 1e6:.0f} MB ceiling for a single Mongo document "
            f"(hard cap {_DOC_LIMIT / 1e6:.0f} MB). The book is too long for "
            f"one-document storage; keep this title local-only."
        )


# -- writes ----------------------------------------------------------------

def push_book(book: Book, checksum: str, chunks: list[Chunk],
              matrix: np.ndarray) -> None:
    """Write one book through to Atlas.

    Never raises into the caller: a failed sync must not discard an ingest
    that already succeeded locally. The book stays serveable either way, it
    just will not survive this container.
    """
    if not enabled():
        return
    try:
        chunk_blob = _pack_chunks(chunks)
        matrix_blob = _pack_matrix(matrix)
        _guard_size(book.id, chunk_blob, matrix_blob)

        database = _database()
        meta = dataclasses.asdict(book)
        meta.pop("id", None)
        meta["_id"] = book.id
        meta["checksum"] = checksum
        database.books.replace_one({"_id": book.id}, meta, upsert=True)
        database.payloads.replace_one(
            {"_id": book.id},
            {"_id": book.id, "chunks": chunk_blob, "matrix": matrix_blob,
             "n_chunks": len(chunks)},
            upsert=True,
        )
        size_mb = (len(chunk_blob) + len(matrix_blob)) / 1e6
        print(f"[atlas] pushed {book.id} ({size_mb:.1f} MB)", flush=True)
    except Exception as exc:                                # noqa: BLE001
        print(f"[atlas] push failed for {book.id}: {exc}", flush=True)


def delete_book(book_id: str) -> None:
    """Mirror a local delete, or the book reappears at the next hydration."""
    if not enabled():
        return
    try:
        database = _database()
        database.books.delete_one({"_id": book_id})
        database.payloads.delete_one({"_id": book_id})
        print(f"[atlas] deleted {book_id}", flush=True)
    except Exception as exc:                                # noqa: BLE001
        print(f"[atlas] delete failed for {book_id}: {exc}", flush=True)


# -- reads -----------------------------------------------------------------

def remote_book_ids() -> list[str]:
    if not enabled():
        return []
    return [d["_id"] for d in _database().books.find({}, {"_id": 1})]


def pull_book(book_id: str):
    """Fetch one book's artifacts as (Book, checksum, chunks, matrix),
    or None if Atlas does not have it."""
    if not enabled():
        return None
    database = _database()
    meta = database.books.find_one({"_id": book_id})
    payload = database.payloads.find_one({"_id": book_id})
    if not meta or not payload:
        return None

    checksum = meta.pop("checksum", "")
    meta.pop("_id", None)
    fields = {f.name for f in dataclasses.fields(Book)}
    book = Book(id=book_id, **{k: v for k, v in meta.items() if k in fields})
    return book, checksum, _unpack_chunks(payload["chunks"]), _unpack_matrix(payload["matrix"])


# -- hydration -------------------------------------------------------------

def hydrate(conn: sqlite3.Connection, force: bool = False) -> int:
    """Rebuild the local index from Atlas. Returns the number of books written.

    Skips books already present locally unless `force`, so a warm disk pays
    only one round trip for the id list. The `app.store.db` import is local
    to avoid an import cycle.
    """
    global _warned
    if not enabled():
        if not _warned:
            print("[atlas] MONGODB_URI unset -- running local-only", flush=True)
            _warned = True
        return 0

    from app.store import db, vectors

    try:
        remote_ids = remote_book_ids()
    except Exception as exc:                                # noqa: BLE001
        # An unreachable Atlas must not stop the API from booting: whatever
        # is already on disk is still perfectly serveable.
        print(f"[atlas] unreachable, serving local index only: {exc}", flush=True)
        return 0

    written = 0
    for book_id in remote_ids:
        if not force and db.get_book(conn, book_id) and vectors.exists(book_id):
            continue
        try:
            pulled = pull_book(book_id)
            if pulled is None:
                continue
            book, checksum, chunks, matrix = pulled
            if not chunks:
                # insert_chunks reads chunks[0].book_id, and an empty book is
                # nothing worth restoring anyway
                print(f"[atlas] skipping {book_id}: no chunks", flush=True)
                continue
            if len(chunks) != len(matrix):
                # both halves are written in one call, so a mismatch means a
                # partial write; refusing beats an unsound prefix
                print(f"[atlas] skipping {book_id}: {len(chunks)} chunks vs "
                      f"{len(matrix)} rows", flush=True)
                continue
            db.upsert_book(conn, book, checksum)
            # upsert_book does not touch edition_pages -- it is owned by
            # set_edition_pages -- so restoring it needs a second call.
            # Without this the reader's declared edition silently resets to
            # our own paging on every redeploy, and their page numbers move.
            if book.edition_pages:
                db.set_edition_pages(conn, book_id, book.edition_pages)
            db.insert_chunks(conn, chunks)      # re-asserts monotonicity
            vectors.save(book_id, matrix)
            written += 1
            print(f"[atlas] hydrated {book_id} ({len(chunks)} chunks)", flush=True)
        except Exception as exc:                            # noqa: BLE001
            print(f"[atlas] hydrate failed for {book_id}: {exc}", flush=True)

    vectors.load.cache_clear()
    return written


def push_local(conn: sqlite3.Connection, force: bool = False) -> int:
    """Upload every locally indexed book. Used by scripts/sync_atlas.py to
    seed Atlas from a machine that has already paid for the embedding."""
    from app.store import db, vectors

    existing = set() if force else set(remote_book_ids())
    pushed = 0
    for book in db.list_books(conn):
        if book.id in existing or not vectors.exists(book.id):
            continue
        row = conn.execute("SELECT checksum FROM books WHERE id = ?",
                           (book.id,)).fetchone()
        push_book(book, row["checksum"] if row else "",
                  db.all_chunks(conn, book.id), vectors.load(book.id))
        pushed += 1
    return pushed
