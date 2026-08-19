"""Re-index every cached book after a chunker or cleaner change.

Positions are offsets into whatever the cleaner decided the text is, so any
change to cleaning or chunking invalidates the existing index. Sources are
already on disk, so nothing is re-downloaded.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import RAW                       # noqa: E402
from app.ingestion.pipeline import ingest_file   # noqa: E402
from app.store import db                         # noqa: E402

if __name__ == "__main__":
    conn = db.connect()
    books = db.list_books(conn)
    print(f"re-indexing {len(books)} book(s)\n")

    for b in books:
        src = next((p for p in RAW.iterdir() if p.stem == b.id
                    or p.stem == "the-glasswing-affair" and b.id.startswith("synth")), None)
        if src is None:
            print(f"  SKIP {b.title[:40]:<42} no cached source")
            continue
        before = b.n_chunks
        try:
            updated = ingest_file(src, book_id=b.id, title=b.title,
                                  author=b.author, source=b.source,
                                  source_url=b.source_url, license=b.license,
                                  force=True)
        except Exception as exc:  # noqa: BLE001
            # one bad book must not abort the rest of the library
            print(f"  FAIL {b.title[:40]:<42} {type(exc).__name__}: {exc}")
            continue
        print(f"  {updated.title[:40]:<42} {before:>4} -> {updated.n_chunks:>4} chunks")

    print("\ndone")
