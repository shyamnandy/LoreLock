"""Generate a synthetic mystery with exact ground-truth reveal positions.

Why this exists: for any real book, a correct answer might come from the
model's memory rather than from retrieval, so a spoiler-leakage measurement
is confounded. This text has never been on the internet. Any correct reveal
is therefore provably a firewall failure, and every fact's position is known
by construction rather than by hand-labelling.

Deterministic -- no LLM, no network. Same seed, same book, same answers.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import RAW, WORDS_PER_PAGE  # noqa: E402

SEED = 20260818
OUT_TXT = RAW / "the-glasswing-affair.txt"
OUT_JSON = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden_novella.json"

CHARACTERS = [
    ("Marcus Vale", "the harbourmaster", "a scar across his left palm"),
    ("Elena Reyes", "the archivist", "a habit of humming when nervous"),
    ("Tobias Crane", "the night watchman", "a limp from an old fall"),
    ("Ilse Fenwick", "the botanist", "ink stains on her cuffs"),
    ("Rufus Adeyemi", "the shipping clerk", "an unfailing memory for dates"),
]

# (fact_key, question, reveal sentence, chapter, NONCE)
#
# Each reveal carries a nonce that appears EXACTLY ONCE in the whole book.
# Ground truth has to be unambiguous: an earlier draft keyed on the culprit's
# first name, which occurs 225 times in ordinary filler, and the evaluation
# duly reported a 17% "leak" that was nothing of the kind. The generator now
# asserts uniqueness rather than trusting it.
PLANTED = [
    ("marcus_role",   "Who is Marcus Vale and what does he do?",
     "Marcus Vale kept the harbour ledgers, a post recorded as wardenscrip in the town roll.",
     1, "wardenscrip"),
    ("elena_meeting", "How do Elena and Marcus know each other?",
     "Elena Reyes had known Marcus Vale since the saltmarrow winter, when both worked the salvage docks.",
     3, "saltmarrow"),
    ("brass_key",     "What is the significance of the brass key?",
     "The brass key opened the kestrelvault, a storeroom the ledgers insisted did not exist.",
     5, "kestrelvault"),
    ("tobias_lie",    "Why did Tobias leave the harbour that night?",
     "Tobias Crane left before midnight because of a tallowbribe pressed into his hand at the gate.",
     8, "tallowbribe"),
    ("culprit",       "Who forged the shipping ledgers?",
     "The forger was the clerk they called Quillhand, and every entry for eleven months was his.",
     12, "quillhand"),
    ("motive",        "Why were the ledgers forged?",
     "The forgeries hid one cargo only: the glasswing orchids raised in the nightglass frames.",
     13, "nightglass"),
]

# Filler must be VARIED. An earlier draft cycled twelve sentences, which made
# every chunk look alike to both retrieval paths and drove recall to noise --
# the fixture was measuring its own uniformity, not the system. Real prose is
# lexically diverse; the generator now composes from wide pools so chunks are
# distinguishable without being informative.
SUBJECTS = ["the tide", "the harbour bell", "a gull", "the ferry", "the crane",
            "the fog", "a lantern", "the customs boat", "the wind", "the rain",
            "a barge", "the lock gate", "the winch", "a rope", "the quay"]
VERBS = ["shifted", "settled", "creaked", "turned", "faded", "rose", "slackened",
         "held", "drifted", "swung", "stalled", "carried", "lifted", "sank"]
PLACES = ["past the breakwater", "along the north wall", "beyond the slipway",
          "under the gantry", "near the coal wharf", "by the weighbridge",
          "across the basin", "behind the customs shed", "at the tide gauge",
          "outside the chandlery", "below the signal mast"]
TIMES = ["before dawn", "at slack water", "after the shift bell", "towards evening",
         "on the turn of the tide", "just after six", "in the last of the light",
         "through the middle watch", "at first light"]
OBJECTS = ["a coil of hemp", "the manifest", "three crates of tin", "a tally stick",
           "the pilot's flag", "a bundle of oilcloth", "the mooring chain",
           "a crate of pitch", "the day book", "a sack of ballast",
           "the signal lamp", "a bill of lading"]
MOODS = ["and nobody remarked on it", "which surprised no one", "as it always did",
         "and the matter rested there", "though it made little difference",
         "and that was the end of it", "without any particular ceremony"]

N_CHAPTERS = 14
WORDS_PER_CHAPTER = 1100


def build() -> tuple[str, dict]:
    rng = random.Random(SEED)
    lines: list[str] = []
    word_count = 0
    reveals: dict[str, int] = {}
    planted_by_chapter: dict[int, list] = {}
    for item in PLANTED:
        planted_by_chapter.setdefault(item[3], []).append(item)

    for ch in range(1, N_CHAPTERS + 1):
        heading = f"CHAPTER {_roman(ch)}"
        lines.append("")
        lines.append(heading)
        lines.append("")
        word_count += len(heading.split())

        chapter_words = 0
        pending = list(planted_by_chapter.get(ch, []))
        # drop the planted sentence roughly midway through its chapter
        drop_at = WORDS_PER_CHAPTER // 2

        while chapter_words < WORDS_PER_CHAPTER:
            if pending and chapter_words >= drop_at:
                key, _q, sentence, _c, _n = pending.pop(0)
                lines.append(sentence)
                reveals[key] = word_count + len(sentence.split())
                word_count += len(sentence.split())
                chapter_words += len(sentence.split())
                continue

            name, role, tell = rng.choice(CHARACTERS)
            sentence = rng.choice([
                f"{rng.choice(SUBJECTS).capitalize()} {rng.choice(VERBS)} "
                f"{rng.choice(PLACES)} {rng.choice(TIMES)}.",
                f"{name} carried {rng.choice(OBJECTS)} {rng.choice(PLACES)}, "
                f"{rng.choice(MOODS)}.",
                f"{name}, {role}, checked {rng.choice(OBJECTS)} {rng.choice(TIMES)}.",
                f"You could always tell {name.split()[0]} by {tell}.",
                f"{rng.choice(SUBJECTS).capitalize()} {rng.choice(VERBS)} "
                f"{rng.choice(TIMES)}, {rng.choice(MOODS)}.",
                f"They left {rng.choice(OBJECTS)} {rng.choice(PLACES)} "
                f"{rng.choice(TIMES)}.",
            ])
            lines.append(sentence)
            word_count += len(sentence.split())
            chapter_words += len(sentence.split())

        # anything not yet placed goes at the chapter's end
        for key, _q, sentence, _c, _n in pending:
            lines.append(sentence)
            reveals[key] = word_count + len(sentence.split())
            word_count += len(sentence.split())

    text = "\n".join(lines).strip() + "\n"
    golden = {
        "book_id": "synth-glasswing",
        "title": "The Glasswing Affair",
        "author": "Synthetic Fixture",
        "total_words": word_count,
        "words_per_page": WORDS_PER_PAGE,
        "questions": [
            {"key": key, "question": q, "reveal_word": reveals[key],
             "reveal_page": reveals[key] // WORDS_PER_PAGE + 1,
             "answer_contains": nonce}
            for key, q, _s, _c, nonce in PLANTED
        ],
    }

    # the fixture is only worth anything if each needle is unique
    lowered = text.lower()
    for _k, _q, _s, _c, nonce in PLANTED:
        count = lowered.count(nonce.lower())
        if count != 1:
            raise AssertionError(
                f"nonce {nonce!r} appears {count} times; ground truth must be "
                f"unambiguous or the leakage metric measures nothing"
            )

    return text, golden


def _roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


if __name__ == "__main__":
    text, golden = build()
    OUT_TXT.write_text(text, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    print(f"wrote {OUT_TXT}  ({golden['total_words']:,} words, "
          f"{golden['total_words'] // WORDS_PER_PAGE} pages)")
    print(f"wrote {OUT_JSON}")
    for q in golden["questions"]:
        print(f"  p{q['reveal_page']:<4} {q['question']}")
