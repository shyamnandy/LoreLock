"""SQLite store: book metadata, the full chunk table, FTS5 keyword index,
sessions and the conversation log.

This file holds STORE B -- the whole book, every position. Nothing that
touches an LLM is allowed to read from here directly; only `app.retrieval`
may, and only through the bounded interfaces it exposes.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence

from app.config import DB_PATH
from app.models import Book, Chunk

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS books (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    author       TEXT NOT NULL,
    source       TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    license      TEXT NOT NULL,
    checksum     TEXT NOT NULL,
    total_words  INTEGER NOT NULL,
    n_chunks     INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'ready',
    -- pages in the reader's own edition; NULL means "use our own paging"
    edition_pages INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    book_id       TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    text          TEXT NOT NULL,
    chapter       INTEGER NOT NULL,
    chapter_title TEXT NOT NULL,
    word_start    INTEGER NOT NULL,
    word_end      INTEGER NOT NULL,
    UNIQUE (book_id, idx)
);
CREATE INDEX IF NOT EXISTS chunks_book_idx ON chunks(book_id, idx);
CREATE INDEX IF NOT EXISTS chunks_book_end ON chunks(book_id, word_end);

-- keyword half of hybrid retrieval. `rowid` mirrors chunks.rowid so the
-- prefix bound and the match live in a single SQL statement.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    book_id    TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL DEFAULT 0,
    n_p        INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ingestion is minutes long, so it cannot be a synchronous request. Jobs
-- live in the DB rather than process memory so any worker can report status.
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id         TEXT PRIMARY KEY,
    query      TEXT NOT NULL,
    book_id    TEXT,
    status     TEXT NOT NULL DEFAULT 'queued',
    message    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position     INTEGER NOT NULL,
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    evidence_ids TEXT NOT NULL,
    prompt_hash  TEXT NOT NULL,
    prompt_text  TEXT NOT NULL,
    ts           TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


_schema_ready = False


def connect(timeout: float = 30.0) -> sqlite3.Connection:
    """Open a connection.

    WAL lets readers run during a write, but a second *writer* still blocks.
    Ingestion holds a write transaction for minutes while embedding, so every
    connection waits rather than failing instantly with "database is locked".
    """
    global _schema_ready
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    if not _schema_ready:
        # DDL takes a write lock; doing it on every connect needlessly
        # contends with a long-running ingest.
        conn.executescript(SCHEMA)
        _migrate(conn)
        _schema_ready = True
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that CREATE TABLE IF NOT EXISTS cannot add to a database
    that already exists. Cheap enough to run on first connect."""
    have = {r["name"] for r in conn.execute("PRAGMA table_info(books)")}
    if "edition_pages" not in have:
        conn.execute("ALTER TABLE books ADD COLUMN edition_pages INTEGER")
        conn.commit()


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------

def upsert_book(conn: sqlite3.Connection, book: Book, checksum: str) -> None:
    conn.execute(
        """INSERT INTO books (id,title,author,source,source_url,license,
                              checksum,total_words,n_chunks)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             title=excluded.title, author=excluded.author,
             source=excluded.source, source_url=excluded.source_url,
             license=excluded.license, checksum=excluded.checksum,
             total_words=excluded.total_words, n_chunks=excluded.n_chunks""",
        (book.id, book.title, book.author, book.source, book.source_url,
         book.license, checksum, book.total_words, book.n_chunks),
    )
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, chunks: Sequence[Chunk]) -> None:
    """Insert chunks, asserting the monotonicity invariant first.

    If this assertion ever fires, the prefix-slice firewall is unsound for
    this book and no amount of downstream filtering would make it safe.
    """
    for a, b in zip(chunks, chunks[1:]):
        if b.idx != a.idx + 1:
            raise ValueError(f"chunk idx not contiguous at {a.idx} -> {b.idx}")
        if b.word_start < a.word_start or b.word_end < a.word_end:
            raise ValueError(
                f"chunk order violates monotonicity at idx {a.idx}: "
                f"({a.word_start},{a.word_end}) -> ({b.word_start},{b.word_end})"
            )

    book_id = chunks[0].book_id
    conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.executemany(
        """INSERT INTO chunks (id,book_id,idx,text,chapter,chapter_title,
                               word_start,word_end)
           VALUES (?,?,?,?,?,?,?,?)""",
        [(c.id, c.book_id, c.idx, c.text, c.chapter, c.chapter_title,
          c.word_start, c.word_end) for c in chunks],
    )
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------

def _row_to_chunk(r: sqlite3.Row) -> Chunk:
    return Chunk(
        id=r["id"], book_id=r["book_id"], idx=r["idx"], text=r["text"],
        chapter=r["chapter"], chapter_title=r["chapter_title"],
        word_start=r["word_start"], word_end=r["word_end"],
    )


def get_book(conn: sqlite3.Connection, book_id: str) -> Book | None:
    r = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
    if r is None:
        return None
    return Book(id=r["id"], title=r["title"], author=r["author"],
                source=r["source"], source_url=r["source_url"],
                license=r["license"], total_words=r["total_words"],
                n_chunks=r["n_chunks"], edition_pages=r["edition_pages"])


def list_books(conn: sqlite3.Connection) -> list[Book]:
    ids = [r["id"] for r in conn.execute("SELECT id FROM books ORDER BY title")]
    return [b for i in ids if (b := get_book(conn, i))]


def all_chunks(conn: sqlite3.Connection, book_id: str) -> list[Chunk]:
    rows = conn.execute(
        "SELECT * FROM chunks WHERE book_id = ? ORDER BY idx", (book_id,)
    ).fetchall()
    return [_row_to_chunk(r) for r in rows]


def chunks_by_ids(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, Chunk]:
    ids = list(ids)
    if not ids:
        return {}
    q = ",".join("?" * len(ids))
    rows = conn.execute(f"SELECT * FROM chunks WHERE id IN ({q})", ids).fetchall()
    return {r["id"]: _row_to_chunk(r) for r in rows}


def n_p_for_position(conn: sqlite3.Connection, book_id: str, position: int) -> int:
    """Number of chunks fully readable at word offset `position`.

    The bound is `word_end <= position`, NOT `word_start <= position`.
    Chunks overlap, so a chunk beginning before the reader position can still
    contain words beyond it. Requiring the chunk to be *entirely* behind the
    reader is the conservative choice, and it is what keeps the eligible set
    an exact prefix of the array.
    """
    r = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE book_id = ? AND word_end <= ?",
        (book_id, position),
    ).fetchone()
    return int(r["n"])


# --------------------------------------------------------------------------
# sessions
#
# A session is one reader's progress through one book. The expensive data
# (chunks, FTS index, embedding matrix) is per-BOOK and shared; only the
# position is per-session. That split is why a thousand readers at a
# thousand positions cost one matrix plus a thousand integers.
#
# `n_p` is deliberately NOT stored. It is derivable from `position`, and a
# stored copy goes stale the moment a book is re-ingested with different
# chunking -- which would silently move the firewall boundary.
# --------------------------------------------------------------------------

def create_session(conn: sqlite3.Connection, session_id: str, book_id: str,
                   position: int = 0) -> dict:
    conn.execute(
        """INSERT INTO sessions (id, book_id, position, n_p) VALUES (?,?,?,0)
           ON CONFLICT(id) DO UPDATE SET book_id=excluded.book_id,
                                         position=excluded.position""",
        (session_id, book_id, position),
    )
    conn.commit()
    return {"id": session_id, "book_id": book_id, "position": position}


def get_session(conn: sqlite3.Connection, session_id: str) -> dict | None:
    r = conn.execute("SELECT id, book_id, position FROM sessions WHERE id = ?",
                     (session_id,)).fetchone()
    return dict(r) if r else None


def set_position(conn: sqlite3.Connection, session_id: str, position: int) -> dict | None:
    if position < 0:
        raise ValueError("position must be non-negative")
    conn.execute("UPDATE sessions SET position = ? WHERE id = ?",
                 (position, session_id))
    conn.commit()
    return get_session(conn, session_id)


def set_edition_pages(conn: sqlite3.Connection, book_id: str,
                      pages: int | None) -> Book | None:
    """Record the page count of the reader's edition, or None to clear it.

    Only the mapping from page to word offset changes -- the text, the chunks
    and the firewall are untouched, so an existing session keeps the exact
    word position it had and merely gets relabelled.
    """
    if pages is not None and pages < 1:
        raise ValueError("pages must be at least 1")
    conn.execute("UPDATE books SET edition_pages = ? WHERE id = ?",
                 (pages, book_id))
    conn.commit()
    return get_book(conn, book_id)


def delete_book(conn: sqlite3.Connection, book_id: str) -> bool:
    """Remove a book and everything derived from it. Returns False if the id
    was not there.

    The cascades declared in the schema are not relied on: `PRAGMA
    foreign_keys` is per-connection and only the connection that ran the
    DDL has it set, so the children are deleted explicitly, deepest first.

    `chunks_fts` is an external-content index, which means deleting rows from
    `chunks` leaves its own rows behind -- a stale docid whose content lookup
    finds nothing, and the keyword half of retrieval would surface a chunk
    the firewall no longer knows about. It is rebuilt, exactly as
    `insert_chunks` does after a re-index.
    """
    if conn.execute("SELECT 1 FROM books WHERE id = ?", (book_id,)).fetchone() is None:
        return False

    with conn:                              # one transaction; all or nothing
        conn.execute(
            """DELETE FROM conversations WHERE session_id IN
                 (SELECT id FROM sessions WHERE book_id = ?)""", (book_id,))
        conn.execute("DELETE FROM sessions WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        # jobs are history, not children -- but a finished job pointing at a
        # book that no longer exists would let the client re-select it
        conn.execute("DELETE FROM ingest_jobs WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
    return True


def session_history(conn: sqlite3.Connection, session_id: str,
                    limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """SELECT position, question, answer, outcome, evidence_ids, ts
             FROM conversations WHERE session_id = ?
         ORDER BY id DESC LIMIT ?""",
        (session_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# ingest jobs
# --------------------------------------------------------------------------

def create_job(conn: sqlite3.Connection, job_id: str, query: str) -> dict:
    conn.execute("INSERT INTO ingest_jobs (id, query, status) VALUES (?,?,'queued')",
                 (job_id, query))
    conn.commit()
    return {"id": job_id, "query": query, "status": "queued",
            "book_id": None, "message": ""}


def update_job(conn: sqlite3.Connection, job_id: str, status: str,
               message: str = "", book_id: str | None = None) -> None:
    conn.execute(
        """UPDATE ingest_jobs
              SET status = ?, message = ?,
                  book_id = COALESCE(?, book_id),
                  updated_at = datetime('now')
            WHERE id = ?""",
        (status, message[:500], book_id, job_id),
    )
    conn.commit()


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    r = conn.execute(
        "SELECT id, query, book_id, status, message FROM ingest_jobs WHERE id = ?",
        (job_id,)).fetchone()
    return dict(r) if r else None


def active_job_for(conn: sqlite3.Connection, query: str) -> dict | None:
    """Avoid launching a second five-minute ingest for the same query."""
    r = conn.execute(
        """SELECT id, query, book_id, status, message FROM ingest_jobs
            WHERE query = ? AND status IN ('queued','running')
         ORDER BY created_at DESC LIMIT 1""", (query,)).fetchone()
    return dict(r) if r else None


def reap_orphan_jobs(conn: sqlite3.Connection) -> int:
    """Fail any job left mid-flight by a process that no longer exists.

    Ingestion runs in a background task owned by one process. If that process
    dies -- a restart, a redeploy, a crash -- the job row stays 'running'
    forever and the UI polls it indefinitely. Reaping at startup is correct
    because nothing else could still be working on it.
    """
    cur = conn.execute(
        """UPDATE ingest_jobs
              SET status = 'failed',
                  message = 'interrupted by a server restart; try again',
                  updated_at = datetime('now')
            WHERE status IN ('queued', 'running')""")
    conn.commit()
    return cur.rowcount
