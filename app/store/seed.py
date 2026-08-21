"""Hydrate an empty DATA_DIR from the seed library baked into the image.

The index is files -- SQLite plus one .npy per book -- and on a free-tier
host those files live on a filesystem that is reset on every deploy and
every wake from sleep. Nothing here makes DATA_DIR durable. It makes the
*starting state* non-empty, so a recycled container comes back with the
same library it shipped with instead of an empty shelf.

Books added at runtime are still lost on a restart. That is a property of
the host, not of this module; the fix for it is a mounted disk.
"""
from __future__ import annotations

import shutil

from app.config import DB_PATH, SEED_DIR, VEC_DIR


def hydrate() -> int:
    """Install the seed library if DATA_DIR has no database yet.

    Returns the number of embedding matrices copied. 0 means there was
    nothing to do, which is the normal case on any host that keeps its disk.

    The trigger is the *absence of the database file*, not an empty books
    table. A reader who deletes every book has made a decision and a restart
    must not quietly undo it; a container that came up with a blank
    filesystem has made no decision at all. Those two states look identical
    through the books table and different on disk, so the disk is what we
    ask.
    """
    seed_db = SEED_DIR / DB_PATH.name
    if DB_PATH.exists() or not seed_db.is_file():
        return 0

    # Matrices first. A book row whose matrix is missing 404s on every
    # question; a matrix with no book row is inert. If the copy is
    # interrupted half way, this ordering leaves the inert failure.
    installed = 0
    seed_vectors = SEED_DIR / "vectors"
    if seed_vectors.is_dir():
        for src in sorted(seed_vectors.glob("*.npy")):
            shutil.copy2(src, VEC_DIR / src.name)
            installed += 1

    # The database last, and via a rename. The health check can fire while
    # this runs, and sqlite reports a half-written file as corrupt rather
    # than absent -- which would be a permanent failure, where "not there
    # yet" is a transient one.
    staged = DB_PATH.with_name(DB_PATH.name + ".seeding")
    shutil.copy2(seed_db, staged)
    staged.replace(DB_PATH)
    return installed
