"""The spoiler firewall.

There is exactly one way to retrieve book content in this codebase, and it
cannot be constructed without a reader position. There is no `search_all`.

Two paths, two different kinds of bound:

  vector   STRUCTURAL -- the operand is `E[:n_p]`; future rows are absent
  keyword  PREDICATE  -- `c.idx < :n_p` in the same SQL statement as MATCH

The keyword path is the one that still relies on a condition being written
correctly, which is precisely why the mutation test in tests/test_firewall.py
targets it.
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np

from app.config import TOP_K
from app.models import Chunk, Evidence, FirewallBreach
from app.store import db, vectors

_WORD = re.compile(r"[A-Za-z0-9']+")


def _fts_query(question: str) -> str:
    """Build a safe FTS5 MATCH expression. User text never reaches FTS5 raw."""
    words = [w for w in _WORD.findall(question) if len(w) > 2]
    return " OR ".join(f'"{w}"' for w in words[:24])


class SafeRetriever:
    """Bounded retrieval over STORE A.

    `position` is a word offset. The eligible set is every chunk lying
    *entirely* behind it (`word_end <= position`) -- chunks overlap, so a
    chunk that merely starts before the reader can still contain words the
    reader has not reached.
    """

    def __init__(self, conn: sqlite3.Connection, book_id: str, position: int):
        if position < 0:
            raise ValueError("position must be non-negative")
        self._conn = conn
        self._book_id = book_id
        self._p = position
        self._n_p = db.n_p_for_position(conn, book_id, position)

    @property
    def n_p(self) -> int:
        return self._n_p

    @property
    def position(self) -> int:
        return self._p

    def n_withheld(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE book_id = ?", (self._book_id,)
        ).fetchone()
        return int(row["n"]) - self._n_p

    # -- paths -------------------------------------------------------------

    def _vector_ids(self, question: str, k: int) -> tuple[list[str], dict[str, float]]:
        """Returns (ranked ids, {id: cosine}). The cosines are the only true
        relevance signal in the system -- RRF scores encode rank alone."""
        if self._n_p == 0:
            return [], {}
        matrix = vectors.prefix(self._book_id, self._n_p)   # STRUCTURAL BOUND
        scores = matrix @ vectors.embed_query(question)
        top = np.argsort(scores)[::-1][:k]
        ids = [f"{self._book_id}::c{int(i):05d}" for i in top]
        return ids, {f"{self._book_id}::c{int(i):05d}": float(scores[i])
                     for i in range(len(scores))}

    def _keyword_ids(self, question: str, k: int) -> list[str]:
        match = _fts_query(question)
        if not match or self._n_p == 0:
            return []
        rows = self._conn.execute(
            """SELECT c.id AS id, bm25(chunks_fts) AS score
                 FROM chunks_fts f
                 JOIN chunks c ON c.rowid = f.rowid
                WHERE chunks_fts MATCH ?
                  AND c.book_id = ?
                  AND c.idx < ?                      -- PREDICATE BOUND
             ORDER BY score ASC
                LIMIT ?""",
            (match, self._book_id, self._n_p, k),
        ).fetchall()
        return [r["id"] for r in rows]

    # -- public ------------------------------------------------------------

    def search(self, question: str, k: int = TOP_K) -> list[Evidence]:
        from app.retrieval.hybrid import fuse

        pool = max(k * 3, 20)
        vec_ids, cosines = self._vector_ids(question, pool)
        fused = fuse(vec_ids, self._keyword_ids(question, pool))[:k]

        selected = self._with_neighbours([cid for cid, _, _ in fused])
        scores = {cid: (s, v) for cid, s, v in fused}

        by_id = db.chunks_by_ids(self._conn, selected)
        evidence = [
            Evidence(
                chunk=by_id[cid],
                score=scores.get(cid, (0.0, "neighbour"))[0],
                via=scores.get(cid, (0.0, "neighbour"))[1],
                similarity=cosines.get(cid, 0.0),
            )
            for cid in selected if cid in by_id
        ]
        # narrative order, not relevance order: a model reading passages in
        # the order the reader met them makes far better sense of them
        evidence.sort(key=lambda e: e.chunk.idx)

        self.assert_safe([e.chunk for e in evidence])
        return evidence

    def _with_neighbours(self, ids: list[str], span: int = 1) -> list[str]:
        """Widen each hit to its immediate neighbours, still inside the prefix.

        Narrative evidence is rarely self-contained: the sentence naming a
        motive often sits in the paragraph after the one that matches the
        query. Pulling idx-1 and idx+1 restores the surrounding exchange.

        The firewall is unaffected -- a neighbour is admitted only if its own
        index is below the boundary, so this can never reach past the reader.
        """
        wanted: set[int] = set()
        for cid in ids:
            idx = int(cid.rsplit("c", 1)[-1])
            for offset in range(-span, span + 1):
                neighbour = idx + offset
                if 0 <= neighbour < self._n_p:      # SAME PREFIX BOUND
                    wanted.add(neighbour)
        return [f"{self._book_id}::c{i:05d}" for i in sorted(wanted)]

    def assert_safe(self, chunks: list[Chunk]) -> None:
        """Re-check the bound before anything is handed to a prompt.

        Raises rather than filters. A silent filter would conceal the very
        bug this system exists to prevent.
        """
        breaches = [c for c in chunks if c.word_end > self._p]
        if breaches:
            raise FirewallBreach(
                f"{len(breaches)} chunk(s) past position {self._p}: "
                + ", ".join(f"{c.id}(end={c.word_end})" for c in breaches[:5])
            )
