import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
"""Build the seed library that ships inside the Docker image.

    python scripts/build_seed.py                # public-domain books only
    python scripts/build_seed.py --list         # show what would be included
    python scripts/build_seed.py --keep ID ...  # override the selection

Reads the working index under DATA_DIR and writes `seed/`, which IS tracked
in git -- that is the whole point, since Render builds from the checkout and
an untracked file cannot reach the image.

WHAT IS DELIBERATELY LEFT OUT
  uploads   The chunks table stores full passage text, so a seed built from
            an upload would commit that book's text to the repository.
            Project Gutenberg is public domain; a reader's own EPUB is not.
  sessions, conversations, ingest_jobs
            One deployment's transcripts are not another's starting state.
  data/raw/ Only /books/{id} DELETE reads it, to clean up an upload's source
            file. Retrieval never touches it, so it is 2 MB of dead weight.
"""
import argparse
import shutil
import sqlite3

from app.config import DB_PATH, ROOT, VEC_DIR

SEED = ROOT / "seed"
# Everything that is not the reader's own copy of something in copyright.
PUBLIC = ("gutenberg", "synthetic")


def _selection(conn: sqlite3.Connection, keep: list[str]) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT id, title, source, n_chunks FROM books ORDER BY id"
    ).fetchall()
    if keep:
        return [r for r in rows if r["id"] in set(keep)]
    return [r for r in rows if r["source"] in PUBLIC]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", nargs="*", default=[],
                    help="book ids to include, overriding the source filter")
    ap.add_argument("--list", action="store_true",
                    help="report the selection and exit without writing")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"no index at {DB_PATH} -- ingest something first")
        return 1

    src = sqlite3.connect(DB_PATH)
    src.row_factory = sqlite3.Row
    everything = src.execute(
        "SELECT id, title, source, n_chunks FROM books ORDER BY id"
    ).fetchall()
    chosen = _selection(src, args.keep)
    ids = {r["id"] for r in chosen}

    for r in everything:
        mark = "keep" if r["id"] in ids else "SKIP"
        print(f"  [{mark}] {r['id']:<68} {r['source']:<10} {r['n_chunks']:>4} chunks")
    if not chosen:
        print("nothing selected")
        src.close()
        return 1
    if args.list:
        src.close()
        return 0

    # VACUUM INTO rather than a file copy: it writes one self-contained file
    # with the WAL already folded in. Copying companion.sqlite3 alone would
    # silently drop whatever was still sitting in companion.sqlite3-wal.
    SEED.mkdir(exist_ok=True)
    staged = SEED / "companion.building"
    staged.unlink(missing_ok=True)
    src.execute("VACUUM INTO ?", (str(staged),))
    src.close()

    out = sqlite3.connect(staged)
    out.execute("PRAGMA foreign_keys = ON")
    marks = ",".join("?" * len(ids))
    out.execute(f"DELETE FROM books WHERE id NOT IN ({marks})", tuple(ids))
    # chunks go by cascade, but chunks_fts is an external-content index: its
    # rows survive their content and would keep matching deleted text.
    out.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    for table in ("conversations", "sessions", "ingest_jobs"):
        out.execute(f"DELETE FROM {table}")
    out.commit()
    out.execute("VACUUM")           # reclaim the deleted book's pages
    out.close()

    final = SEED / DB_PATH.name
    staged.replace(final)

    seed_vectors = SEED / "vectors"
    if seed_vectors.exists():
        shutil.rmtree(seed_vectors)     # stale matrices are worse than none
    seed_vectors.mkdir(parents=True)
    missing = []
    for book_id in sorted(ids):
        matrix = VEC_DIR / f"{book_id}.npy"
        if matrix.exists():
            shutil.copy2(matrix, seed_vectors / matrix.name)
        else:
            missing.append(book_id)

    size = final.stat().st_size + sum(
        p.stat().st_size for p in seed_vectors.glob("*.npy"))
    print(f"\nwrote {final.relative_to(ROOT)} + {len(ids) - len(missing)} "
          f"matrices  ({size / 1e6:.1f} MB)")
    if missing:
        # A book row without a matrix answers 404 to every question, so this
        # is a broken seed, not a warning to file away.
        print(f"MISSING MATRICES: {', '.join(missing)} -- re-ingest and rebuild")
        return 1
    print("commit seed/ -- Render builds from the checkout, not your disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
