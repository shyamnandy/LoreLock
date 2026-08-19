"""Prompt text, kept in one place so the eval harness can diff it over time."""

ANSWER_SYSTEM = """\
You are a reading companion for someone part-way through a book.

You will be given numbered evidence passages. They are the ONLY things you \
may use. You have no other knowledge of this book -- if you believe you \
recognise it, that recollection is off-limits and very likely concerns \
events the reader has not reached yet.

The passages are given in narrative order, and some are included only as \
context around a match, so not all of them will be relevant.

Rules, in order of priority:
1. Every factual sentence must end with a citation of the passage that \
supports it, in square brackets, using the SHORT id exactly as shown at the \
start of that passage -- for a passage headed [chunk abc::c00042] you write \
[c00042]. One id per bracket. Never invent an id, and never copy the example.
2. If the passages do not support an answer, reply with exactly: INSUFFICIENT
3. Never speculate about what happens later, never foreshadow, never hint \
that something is significant later on.
4. ANSWER IN YOUR OWN WORDS. Explain what happened and why. Do not quote or \
paste the passages back -- a reply that is mostly copied text is a failure, \
even if the citation is correct.
5. Name people and places explicitly instead of "he" or "she". The reader is \
asking precisely because they have lost track of who is who.
6. At most four sentences. Plain, warm, direct. No preamble, no "based on \
the passages".
"""

ANALYZER_SYSTEM = """\
Classify the reader's question into exactly one intent. Reply with the \
single lowercase word and nothing else.

character     -- who someone is
event         -- what happened
motivation    -- why someone acted
relationship  -- how two people relate
summary       -- recap of the story so far
clue          -- significance of a detail or object
prediction    -- what will happen next
"""

VALIDATOR_SYSTEM = """\
You check a draft answer against the passages it cites.

The draft SHOULD paraphrase rather than quote. Rewording, summarising, and \
combining several passages into one sentence are all correct behaviour and \
are NOT grounds for rejection. Judge only whether the substance of each \
claim is present in the passages.

Reply with exactly one word:
GROUNDED   -- every claim is supported by the cited passages, in any wording
UNGROUNDED -- some claim states a fact that is simply not in them
"""


def fit_to_budget(evidence, max_chars: int):
    """Drop the weakest passages until the payload fits a token budget.

    Neighbour expansion can turn 8 matches into 21 passages, which on Groq's
    free tier (8,000 TPM) means an HTTP 413 and no answer at all. Keep the
    highest-scoring ones, then restore narrative order -- a model reading
    passages in the order the reader met them makes better sense of them
    than one reading them by rank.

    Ranking is structural, not by score. Seeds are what retrieval actually
    chose; neighbours are context around them. Ranking by cosine drops the
    passage where one revealing sentence sits in 300 words of scene-setting
    -- exactly the passage worth keeping -- because its embedding is
    dominated by the filler around it.
    """
    seeds = [e for e in evidence if e.via != "neighbour"]
    neighbours = [e for e in evidence if e.via == "neighbour"]

    def cost(e) -> int:
        return len(e.chunk.text) + 60          # + the citation header

    kept, used = [], 0
    for e in sorted(seeds, key=lambda e: e.score, reverse=True):
        if kept and used + cost(e) > max_chars:
            continue
        kept.append(e)
        used += cost(e)

    # then spend what is left on context, nearest to a kept seed first
    seed_idx = {e.chunk.idx for e in kept}
    for e in sorted(neighbours,
                    key=lambda e: min((abs(e.chunk.idx - i) for i in seed_idx),
                                      default=10**6)):
        if used + cost(e) > max_chars:
            continue
        kept.append(e)
        used += cost(e)

    return sorted(kept, key=lambda e: e.chunk.idx)


def build_answer_prompt(question: str, evidence) -> str:
    blocks = "\n\n".join(
        f"[chunk {e.chunk.id}] (chapter {e.chunk.chapter}, page {e.chunk.page})\n"
        f"{e.chunk.text}"
        for e in evidence
    )
    return f"Evidence passages:\n\n{blocks}\n\nReader's question: {question}"
