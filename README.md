# Spoiler-Safe Reading Companion

A temporal RAG agent that answers questions about a book using **only the
part the reader has already read**. The spoiler boundary is enforced by the
shape of the data, not by instructing a model.

> Ask your book anything without spoiling what comes next.

---

## The one idea

Chunks are stored in narrative order, so "everything the reader has read" is
a **contiguous prefix** of an array. A session's searchable matrix is the
slice `E[:n_p]`.

```python
matrix = vectors.prefix(book_id, n_p)      # shape == (n_p, 384)
scores = matrix @ vectors.embed_query(question)
```

There is no `WHERE position <= P` to forget on the vector path, because
future rows are not in the operand. A similarity search *cannot* return
chunk `n_p`; it does not exist in the array being searched.

The keyword path is the one place a predicate is still required
(`c.idx < :n_p`, in the same SQL statement as the `MATCH`), which is exactly
why `tests/test_firewall.py` contains a mutation test that removes it and
proves the suite goes red.

### The bound is `word_end`, not `word_start`

Chunks overlap by 40 words. A chunk that *starts* before the reader can
still contain words beyond them, so eligibility requires the chunk to lie
entirely behind the reader:

```sql
SELECT COUNT(*) FROM chunks WHERE book_id = ? AND word_end <= ?
```

This is conservative by one chunk and keeps the eligible set an exact prefix.

---

## Architecture

```
                       question + position
                                |
                       analyze_question            (agent, cheap model)
                                |
                          retrieve                 SafeRetriever
                                |                  vector: E[:n_p]  STRUCTURAL
                                |                  keyword: idx<n_p PREDICATE
                                |                  fused with RRF
                        assess_evidence
                          /            \
              sufficient /              \ thin
                        /                \
                   answer            spoiler_oracle      STORE B, air-gapped
                      |                 /        \        returns one bool
                  validate       refuse_spoiler  refuse_thin
                   |     ^
        grounded   |     | uncited, retry <= 2
                   v     |
                  END <--+
```

Two stores, and a strict rule about what may cross between them:

| Store | Contents | Reachable by |
|---|---|---|
| **A** | chunks with `word_end <= P` | the answer path |
| **B** | the full book, all positions | `app/retrieval/oracle.py` only, which returns a boolean and never text |

The oracle is deliberately **not** an LLM call, so there is no code path by
which future prose enters a prompt.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate           # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python scripts/ingest.py "The Adventures of Sherlock Holmes"
python scripts/demo.py                    # the four-beat demo
```

Then the UI — the API and the web client are separate processes:

```bash
.venv/Scripts/python.exe -m uvicorn app.api:api --port 8000
cd web && npm install && npm run dev       # http://localhost:3000
```

No API key is required to run the firewall, the tests, or the demo: the
default `stub` provider extracts and cites a supplied passage, so the whole
graph — including the citation checker — runs offline.

For real answers, set a provider:

```bash
export LLM_PROVIDER=gemini   GEMINI_API_KEY=...      # or
export LLM_PROVIDER=groq     GROQ_API_KEY=...        # or
export LLM_PROVIDER=ollama                           # local, no key
```

---

## Tests

```bash
python -m pytest -q
```

| File | Covers |
|---|---|
| `test_ingestion.py` | `idx` monotonic in both offsets; chapter continuity; the `word_end` bound |
| `test_firewall.py` | prefix identity, per-position bounds, shingle leak sweep, FTS5 injection, **the mutation test** |
| `test_graph.py` | the three outcomes, prompt payload contains no future text, citations subset of evidence |

The mutation test is the one worth reading. A firewall test that still
passes when you break the firewall proves nothing.

---

## Evaluation

Real books are confounded: the model has already read *Dracula*, so a
correct answer might come from memory rather than retrieval. So the eval
runs against a synthetic novella with facts planted at known positions:

```bash
python scripts/make_novella.py       # 15,502 words, 6 facts, exact positions
python scripts/evaluate.py --empty-probe
```

| Metric | Meaning |
|---|---|
| spoiler leakage rate | reveal text appearing in an answer given before its position |
| prompt leakage rate | reveal text reaching the prompt at all — **must be 0%** |
| safe refusal accuracy | future-dependent questions correctly refused |
| post-reveal answer rate | the same question answered once the reader passes the reveal |
| parametric leakage | correct reveals with **zero evidence supplied** — measures the model answering from memory |

The last one is the honest measurement: the firewall cannot prevent a model
from knowing a famous book, only from being *fed* it. Reporting the number
is the credible position.

---

## Layout

```
app/
  config.py            all tunables
  models.py            Chunk, Book, Evidence, FirewallBreach
  graph.py             LangGraph: 2 conditional branches, 1 cycle, typed state
  ingestion/           sources (Gutendex) · parsers · cleaner · chunker · pipeline
  retrieval/           firewall (SafeRetriever) · hybrid (RRF) · oracle (Store B)
  store/               db (SQLite + FTS5) · vectors (NumPy prefix slice)
  agents/              llm (gemini|groq|ollama|stub) · analyzer · validator · prompts
tests/                 35 tests, no network, no API key
scripts/               ingest · make_novella · evaluate · demo
web/                   React (Vite) client; the position slider is the demo
```

---

## Two findings worth knowing

**A cheap embedding statistic cannot judge sufficiency.** The first design
thresholded the RRF fused score to decide whether evidence was good enough
to answer from. RRF encodes *rank only* -- its top hit scores the same
whether the passage is on point or unrelated -- so the gate passed
everything and the spoiler branch was unreachable. Switching to raw cosine
was better but still corpus-dependent: `bge-small` scores 0.455 for "what is
the capital of Peru?" against *Sherlock Holmes*, and on a book whose prose is
uniform in subject matter the off-topic control out-scored the real
questions. A z-score against the eligible corpus did not separate them
either.

So sufficiency is judged where the judgement is actually possible: the
answer agent reads the passages and replies `INSUFFICIENT`, and only then
does the oracle decide which refusal the reader gets. `SUFFICIENCY_FLOOR`
survives as a cheap screen for degenerate cases, nothing more.

**Ground truth has to be unambiguous.** The first novella keyed the culprit
fact on the character's first name, which appears 225 times in ordinary
filler. The evaluation duly reported a 17% "prompt leak" that was nothing of
the kind -- the firewall had held perfectly. Each planted fact now carries a
nonce that the generator asserts appears exactly once in the whole book.

## Integration bugs worth knowing about

Wiring a real provider surfaced eight failures, most of which presented as
plausible-looking spoiler-safe behaviour rather than as errors:

| Symptom | Actual cause |
|---|---|
| `403 error code: 1010` with a valid key | Cloudflare blocks urllib's default `Python-urllib/3.x` User-Agent |
| `model_not_found` | Groq withdrew the llama-3.x chat models; `scripts/models.py` lists what is live |
| `413 Payload Too Large` | neighbour expansion turned 5 matches into 21 passages, over the 8,000 TPM cap |
| every answer "not enough evidence" | provider errors were caught and returned as `INSUFFICIENT` -- an outage scored as 100% refusal accuracy |
| correct answers refused | the model-based validator saw evidence truncated to 400 chars, ~23% of a chunk, and judged the claims unsupported |
| correct answers refused | citations arrived as `[c00042]` or `[c1, c2]`; the parser demanded `[book::c00042]` |
| citations rejected as foreign | the prompt's example `[bk::c00042]` taught the model to write `bk::` as the prefix |
| blank answer, outcome `answered` | `gpt-oss` models reason before answering and can spend the whole budget on `reasoning`, returning empty `content`; an empty draft then validated as grounded, since it makes no unsupported claims |

Two of these are worth generalising. A provider failure must never be
convertible into a refusal -- `outcome: "error"` exists precisely so an
outage cannot masquerade as correct behaviour. And vacuous truth is not
grounding: an empty answer has no unsupported claims, and an earlier
validator happily passed it.

## Free-tier arithmetic

Groq's free tier allows **8,000 tokens per minute and 200,000 per day, per
model**. At ~2,200 tokens per answer call that is roughly three turns a
minute, and the day's budget is gone in ~45 turns of development. Practical
consequences:

- `MAX_EVIDENCE_CHARS` is 9,000 (~2,250 tokens), not the 18,000 that felt
  generous -- a 4,000-token prompt permits barely one call a minute.
- `VALIDATOR_USE_MODEL=0` halves token use by skipping the second call. The
  deterministic citation check is the load-bearing layer and always runs.
- Per-node routing spreads load across separate quota buckets, since the
  limits are per model.
- Rate-limit retries honour the provider's own `retry-after` rather than
  guessing.

## Known limits

- **The refusal leaks about one bit.** "I can't answer without spoiling"
  tells you the thing *does* get resolved. Both refusal branches render
  identically in the UI; they are distinguished only internally, for eval.
- **Pages are synthetic** — 250 words each, tied to the indexed edition.
  Cross-edition alignment is a later feature.
- **Scattered evidence degrades.** Answers assembled across many chapters
  retrieve poorly. That is a reranking problem, not a model-size problem.
- **Parametric leakage is reduced, not eliminated.** You get a number, not
  a guarantee.
- **Retrieval recall on the novella is 50%.** A one-sentence fact inside a
  200-word chunk is diluted, so the reveal chunk does not always rank into
  the top 8 even when eligible. Reported separately from leakage precisely
  because it is a ranking problem, not a firewall problem -- reranking or
  smaller chunks would move it.
- **The `stub` provider cannot judge relevance.** It quotes the best
  lexically-matching passage, which is enough to exercise every branch
  offline but not to produce a convincing demo. Set a real provider for
  that; the firewall and its tests are identical either way.
