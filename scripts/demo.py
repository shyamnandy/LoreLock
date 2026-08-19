"""The four-beat demo.

    python scripts/demo.py [book_id]

Beat 3 is the one that lands: the identical question, the identical prompt
template, the identical model -- only the slice grew.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agents import llm                       # noqa: E402
from app.config import WORDS_PER_PAGE            # noqa: E402
from app.graph import ask                        # noqa: E402
from app.store import db, vectors                # noqa: E402

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"


def beat(n: int, title: str) -> None:
    print(f"\n{BOLD}DEMO {n}{OFF}  {title}\n{DIM}{'-' * 68}{OFF}")


def turn(book, question: str, page: int) -> dict:
    out = ask("demo", book.id, question, position=page * WORDS_PER_PAGE)
    icon = {"answered": "[answered]", "refused_spoiler": "[spoiler-refused]",
            "refused_thin": "[thin-refused]"}[out["outcome"]]
    print(f"  page {page:<5} {icon}")
    print(f"  Q  {question}")
    print(f"  A  {out['answer'][:320]}")
    print(f"  {DIM}eligible {out.get('n_p', 0)} chunks · withheld "
          f"{out.get('n_withheld', 0)} · intent {out.get('intent', '-')}{OFF}")
    return out


if __name__ == "__main__":
    conn = db.connect()
    ready = [b for b in db.list_books(conn) if vectors.exists(b.id)]
    if not ready:
        sys.exit("No indexed book. Run: python scripts/ingest.py \"Dracula\"")

    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    book = next((b for b in ready if b.id == wanted), ready[0])

    print(f"\n{BOLD}{book.title}{OFF} — {book.author}")
    print(f"{DIM}{book.total_words:,} words · {book.total_pages} pages · "
          f"{book.n_chunks} chunks · provider {llm.provider_name()}{OFF}")

    early, late = 40, min(book.total_pages, 320)
    late_q = "Who forged the shipping ledgers?" if book.id.startswith("synth") \
        else "Who is Irene Adler and why does Holmes remember her?"
    early_q = "Who is the narrator and what does he do?"

    beat(1, "a question the reader has already passed")
    turn(book, early_q, early)

    beat(2, "a question whose answer is revealed later")
    turn(book, late_q, early)

    beat(3, "the same question, further along")
    turn(book, late_q, late)

    beat(4, "why it held")
    row = conn.execute("SELECT prompt_text FROM conversations WHERE session_id='demo' "
                       "ORDER BY id DESC LIMIT 1").fetchone()
    print("  The model never received a future passage. The firewall runs")
    print("  before retrieval, and the prompt is logged so you can check:")
    print(f"  {DIM}last prompt: {len(row['prompt_text'])} chars, "
          f"{row['prompt_text'].count('[chunk ')} passages{OFF}")
