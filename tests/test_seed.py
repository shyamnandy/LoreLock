"""The seed hydration that keeps a diskless deploy from booting empty.

These are filesystem tests, not retrieval tests: the question is only ever
"did the right bytes land in DATA_DIR, and did they land in an order that
cannot leave a half-installed library behind".
"""
from __future__ import annotations

import numpy as np
import pytest

from app.store import seed


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """A blank DATA_DIR and a populated SEED_DIR, wired into the module."""
    data, seed_dir = tmp_path / "data", tmp_path / "seed"
    (data / "vectors").mkdir(parents=True)
    (seed_dir / "vectors").mkdir(parents=True)

    db_path = data / "companion.sqlite3"
    monkeypatch.setattr(seed, "DB_PATH", db_path)
    monkeypatch.setattr(seed, "VEC_DIR", data / "vectors")
    monkeypatch.setattr(seed, "SEED_DIR", seed_dir)

    (seed_dir / "companion.sqlite3").write_bytes(b"SQLite format 3\x00seeded")
    np.save(seed_dir / "vectors" / "bookA.npy", np.zeros((3, 4), dtype=np.float32))
    np.save(seed_dir / "vectors" / "bookB.npy", np.zeros((5, 4), dtype=np.float32))
    return db_path, data / "vectors", seed_dir


def test_hydrates_a_blank_data_dir(paths):
    db_path, vec_dir, _ = paths
    assert seed.hydrate() == 2
    assert db_path.read_bytes().endswith(b"seeded")
    assert {p.name for p in vec_dir.glob("*.npy")} == {"bookA.npy", "bookB.npy"}


def test_is_idempotent(paths):
    seed.hydrate()
    # second boot on a host that kept its disk: nothing to do, and nothing
    # touched -- re-copying would silently revert a re-ingest
    assert seed.hydrate() == 0


def test_leaves_an_existing_library_alone(paths):
    """A reader who deleted every book has made a decision. An empty books
    table is not the same state as an absent file, and only the second one
    means "this container came up blank"."""
    db_path, _, _ = paths
    db_path.write_bytes(b"SQLite format 3\x00the reader's own, emptied")
    assert seed.hydrate() == 0
    assert db_path.read_bytes().endswith(b"emptied")


def test_no_seed_is_not_an_error(paths):
    """A host with a mounted disk ships no seed. That is a configuration,
    not a failure, so boot must not raise."""
    _, _, seed_dir = paths
    (seed_dir / "companion.sqlite3").unlink()
    assert seed.hydrate() == 0


def test_database_lands_after_the_matrices(paths, monkeypatch):
    """Ordering is the whole safety argument: interrupted after the matrices
    you get an inert extra file, interrupted before them you get book rows
    whose embeddings are missing -- a 404 on every question."""
    db_path, vec_dir, _ = paths
    seen: list[str] = []
    real_copy = seed.shutil.copy2

    def spy(src, dst):
        seen.append("db" if str(src).endswith(".sqlite3") else "matrix")
        return real_copy(src, dst)

    monkeypatch.setattr(seed.shutil, "copy2", spy)
    seed.hydrate()
    assert seen == ["matrix", "matrix", "db"]
