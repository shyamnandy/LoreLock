import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
"""CLI: python scripts/ingest.py "The Adventures of Sherlock Holmes" """
import sys
from app.ingestion.pipeline import ingest_from_gutenberg
from app.store import db

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "The Adventures of Sherlock Holmes"
    book = ingest_from_gutenberg(query)
    conn = db.connect()
    chunks = db.all_chunks(conn, book.id)
    print(f"{book.title} — {book.author}")
    print(f"  id          {book.id}")
    print(f"  words       {book.total_words:,}   pages {book.total_pages}")
    print(f"  chunks      {book.n_chunks}")
    chapters = sorted({c.chapter for c in chunks})
    print(f"  chapters    {len(chapters)}")
    for c in chunks[:2]:
        print(f"    idx={c.idx} ch={c.chapter} p{c.page} [{c.word_start},{c.word_end})")
        print(f"      {c.text[:90]}...")
