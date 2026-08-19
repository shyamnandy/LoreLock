"""Reader memory: a position-bounded record of what the reader knows.

The invariant is enforced on WRITE, not on read. A fact whose supporting
chunk lies past the reader position is refused at the door -- there is no
path by which a future fact enters the store and is filtered out later.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from app.models import Evidence, FirewallBreach


@dataclass(frozen=True, slots=True)
class MemoryFact:
    fact: str
    source_chunk_ids: list[str]
    max_position: int


def remember(conn: sqlite3.Connection, session_id: str, fact: str,
             evidence: list[Evidence], reader_position: int) -> MemoryFact:
    if not evidence:
        raise ValueError("a memory fact must cite at least one chunk")

    max_position = max(e.chunk.word_end for e in evidence)
    if max_position > reader_position:
        raise FirewallBreach(
            f"refusing to store a fact sourced at {max_position}, "
            f"past reader position {reader_position}"
        )

    ids = [e.chunk.id for e in evidence]
    conn.execute(
        """INSERT INTO memory_facts (session_id, fact, source_chunk_ids, max_position)
           VALUES (?,?,?,?)""",
        (session_id, fact, ",".join(ids), max_position),
    )
    conn.commit()
    return MemoryFact(fact, ids, max_position)


def recall(conn: sqlite3.Connection, session_id: str,
           reader_position: int) -> list[MemoryFact]:
    rows = conn.execute(
        """SELECT fact, source_chunk_ids, max_position FROM memory_facts
            WHERE session_id = ? AND max_position <= ?
         ORDER BY max_position""",
        (session_id, reader_position),
    ).fetchall()
    return [MemoryFact(r["fact"], r["source_chunk_ids"].split(","), r["max_position"])
            for r in rows]
