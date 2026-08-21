"""End-to-end ingestion: acquire -> parse -> clean -> section -> chunk ->
embed -> persist. Runs once per book; a cache hit skips the whole thing."""
from __future__ import annotations

import hashlib
from pathlib import Path

from app.ingestion import cleaner, parsers, sources
from app.ingestion.chunker import Section, chunk_book, total_words
from app.models import Book
from app.store import db, vectors


def _sections_from_text(raw: str) -> list[Section]:
    text = cleaner.normalise(cleaner.strip_gutenberg_boilerplate(raw))
    return cleaner.split_into_sections(text)


def ingest_file(
    path: Path,
    book_id: str,
    title: str,
    author: str,
    source: str = "upload",
    source_url: str = "",
    license: str = "user-supplied",
    force: bool = False,
) -> Book:
    conn = db.connect()
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    existing = db.get_book(conn, book_id)
    if existing and vectors.exists(book_id) and not force:
        row = conn.execute("SELECT checksum FROM books WHERE id = ?",
                           (book_id,)).fetchone()
        # a cache hit is only a hit if the source bytes still match; otherwise
        # positions in the index no longer refer to the text on disk
        if row and row["checksum"] == checksum:
            return existing

    raw = parsers.parse(path)
    sections = _sections_from_text(raw)
    chunks = chunk_book(book_id, sections)
    if not chunks:
        raise ValueError(f"no text extracted from {path}")

    book = Book(
        id=book_id, title=title, author=author, source=source,
        source_url=source_url, license=license,
        total_words=total_words(sections), n_chunks=len(chunks),
    )
    db.upsert_book(conn, book, checksum)
    db.insert_chunks(conn, chunks)          # asserts monotonicity
    matrix = vectors.embed_texts([c.text for c in chunks], progress=True)
    vectors.save(book_id, matrix)
    vectors.load.cache_clear()

    return book


def ingest_from_gutenberg(query: str, force: bool = False) -> Book:
    candidates = sources.search(query, limit=5)
    if not candidates:
        raise LookupError(f"no permitted source found for {query!r}")
    best = max(candidates, key=lambda c: c.download_count)
    path, _ = sources.download(best)
    return ingest_file(
        Path(path), book_id=best.book_id, title=best.title, author=best.author,
        source=best.source, source_url=best.url, license=best.license,
        force=force,
    )
