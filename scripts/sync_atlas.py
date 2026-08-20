"""Push the local index up to Atlas, or pull it back down.

Embedding a book takes minutes and a 512 MB instance is the worst place to
spend them. Ingest locally, run this once, and every deploy afterwards
hydrates in seconds.

    python scripts/sync_atlas.py push          # local  -> Atlas (skips existing)
    python scripts/sync_atlas.py push --force  # local  -> Atlas (overwrite)
    python scripts/sync_atlas.py pull          # Atlas  -> local (skips existing)
    python scripts/sync_atlas.py status        # what is where

Needs MONGODB_URI in the environment or in .env.
"""
from __future__ import annotations

import pathlib
import sys

# run as a script, so the repo root is not on sys.path (matches scripts/ingest.py)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from contextlib import closing                               # noqa: E402

from app.store import db, remote, vectors                    # noqa: E402


def _status() -> int:
    with closing(db.connect()) as conn:
        local = {b.id for b in db.list_books(conn) if vectors.exists(b.id)}
    try:
        upstream = set(remote.remote_book_ids())
    except Exception as exc:                                # noqa: BLE001
        print(f"atlas unreachable: {exc}")
        return 1

    print(f"local: {len(local)} book(s), atlas: {len(upstream)} book(s)\n")
    for book_id in sorted(local | upstream):
        here = "local" if book_id in local else "  -  "
        there = "atlas" if book_id in upstream else "  -  "
        print(f"  {here}  {there}  {book_id}")
    if missing := local - upstream:
        print(f"\n{len(missing)} book(s) not yet in Atlas -- run: "
              f"python scripts/sync_atlas.py push")
    return 0


def main() -> int:
    if not remote.enabled():
        print("MONGODB_URI is not set. Put it in .env or export it, e.g.\n"
              "  MONGODB_URI=mongodb+srv://user:pass@cluster.xxxxx.mongodb.net/")
        return 2

    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    force = "--force" in sys.argv

    if command == "status":
        return _status()

    if command == "push":
        with closing(db.connect()) as conn:
            pushed = remote.push_local(conn, force=force)
        print(f"pushed {pushed} book(s)"
              + ("" if pushed else " -- everything was already there"))
        return 0

    if command == "pull":
        with closing(db.connect()) as conn:
            got = remote.hydrate(conn, force=force)
        print(f"hydrated {got} book(s)")
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
