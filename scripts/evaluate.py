"""Spoiler-leakage evaluation.

Runs the golden question set against the synthetic novella at positions
either side of each fact's reveal, and reports the metrics the design
promises. Every number here is measured, not asserted.

    python scripts/evaluate.py [--empty-probe]
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agents import llm                                    # noqa: E402
from app.agents.prompts import ANSWER_SYSTEM                  # noqa: E402
from app.graph import ask                                     # noqa: E402
from app.ingestion.pipeline import ingest_file                # noqa: E402
from app.config import RAW                                    # noqa: E402

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden_novella.json"
MARGIN = 400   # words either side of the reveal to sample


def ensure_book(golden: dict):
    return ingest_file(
        RAW / "the-glasswing-affair.txt",
        book_id=golden["book_id"], title=golden["title"], author=golden["author"],
        source="synthetic", license="generated fixture",
    )


def run(golden: dict) -> dict:
    book = ensure_book(golden)
    rows = []

    for q in golden["questions"]:
        reveal = q["reveal_word"]
        before = max(0, reveal - MARGIN)
        after = min(book.total_words, reveal + MARGIN)

        b = ask("eval-before", book.id, q["question"], position=before)
        a = ask("eval-after", book.id, q["question"], position=after)

        needle = q["answer_contains"].lower()
        # retrieval recall is reported separately from answer quality: a miss
        # here is a ranking problem, not a firewall or a model problem, and
        # conflating the two hides which layer needs work.
        retrieved = any(needle in e.chunk.text.lower()
                        for e in a.get("evidence", []))
        rows.append({
            "question": q["question"],
            "reveal_page": q["reveal_page"],
            "before": {
                "outcome": b["outcome"],
                "leaked": needle in b["answer"].lower(),
                "evidence": len(b.get("evidence", [])),
                "prompt_leaked": needle in b.get("prompt_text", "").lower(),
            },
            "after": {
                "outcome": a["outcome"],
                "retrieved": retrieved,
                "answered_correctly": needle in a["answer"].lower(),
                "evidence": len(a.get("evidence", [])),
            },
        })

    n = len(rows)
    leaks = sum(r["before"]["leaked"] for r in rows)
    prompt_leaks = sum(r["before"]["prompt_leaked"] for r in rows)
    refused = sum(r["before"]["outcome"].startswith("refused") for r in rows)
    recovered = sum(r["after"]["answered_correctly"] for r in rows)
    recalled = sum(r["after"]["retrieved"] for r in rows)

    return {
        "rows": rows,
        "metrics": {
            "questions": n,
            "spoiler_leakage_rate": leaks / n,
            "prompt_leakage_rate": prompt_leaks / n,
            "safe_refusal_accuracy": refused / n,
            "post_reveal_retrieval_recall": recalled / n,
            "post_reveal_answer_rate": recovered / n,
        },
    }


def empty_probe(golden: dict) -> dict:
    """Parametric leakage, measured in isolation: no evidence at all."""
    hits = 0
    for q in golden["questions"]:
        try:
            reply = llm.chat(ANSWER_SYSTEM, f"Evidence passages:\n\n(none)\n\n"
                                            f"Reader's question: {q['question']}")
            text = reply.text.lower()
        except llm.LLMError as exc:
            text = f"(provider error: {exc})"
        if q["answer_contains"].lower() in text:
            hits += 1
    return {"provider": llm.provider_name(),
            "parametric_leakage_rate": hits / len(golden["questions"])}


if __name__ == "__main__":
    golden = json.loads(GOLDEN.read_text())
    result = run(golden)

    print(f"\n  spoiler-leakage evaluation — {golden['title']}")
    print(f"  provider: {llm.provider_name()}\n")
    head = (f"  {'reveal':>7}  {'before':<16} {'leak':<5} {'after':<16} "
            f"{'got':<4} {'ok':<3}  question")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in result["rows"]:
        print(f"  p{r['reveal_page']:<6} {r['before']['outcome']:<16} "
              f"{'YES' if r['before']['leaked'] else '.':<5} "
              f"{r['after']['outcome']:<16} "
              f"{'y' if r['after']['retrieved'] else '.':<4} "
              f"{'y' if r['after']['answered_correctly'] else '.':<3}  "
              f"{r['question'][:44]}")

    m = result["metrics"]
    print(f"\n  spoiler leakage rate      {m['spoiler_leakage_rate']:.0%}")
    print(f"  prompt leakage rate       {m['prompt_leakage_rate']:.0%}   <- must be 0%")
    print(f"  safe refusal accuracy     {m['safe_refusal_accuracy']:.0%}")
    print(f"  post-reveal retrieval     {m['post_reveal_retrieval_recall']:.0%}"
          f"   <- ranking, not firewall")
    print(f"  post-reveal answer rate   {m['post_reveal_answer_rate']:.0%}"
          f"   <- depends on the provider")

    if "--empty-probe" in sys.argv:
        p = empty_probe(golden)
        print(f"\n  parametric leakage ({p['provider']})  "
              f"{p['parametric_leakage_rate']:.0%}")
