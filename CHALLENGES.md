# Challenges

A running log of problems hit while building and deploying LoreLock, with the
diagnosis and the fix. Written down because most of them were *not* what they
first looked like, and the misleading first impression is the part worth
remembering.

---

## 1. Commits did not count toward GitHub contributions

**Symptom.** Three commits were pushed and visible in the repo, but the
contribution graph stayed empty.

**First guess (wrong).** That the configured email was the wrong Gmail address.

**Actual cause.** `user.email` was `shyam@example.com`, set both locally and
globally. `example.com` is an IANA reserved domain, so no GitHub account can
ever verify it. GitHub links a commit to an account purely by author email; an
unmatched address renders as a plain name with no profile and counts for
nothing.

**Fix.** Set `user.email` to a *GitHub-verified* address, rewrite the three
existing commits, force-push.

```bash
git config --global user.email "you@verified.example"
git rebase --root --exec 'git commit --amend --no-edit --reset-author'
git push --force-with-lease
```

**Two traps inside the fix.**

- `--reset-author` also resets the **author date** to now. All three commits
  collapsed onto a single day, which would have squashed the contribution graph
  into one square — the opposite of the goal. The dates had to be restored
  explicitly with `--date` plus `GIT_COMMITTER_DATE`.
- The rebase aborted on `web/tsconfig.tsbuildinfo`: a TypeScript build cache
  that is gitignored *now* but was committed in the first commit, so checking
  out that commit would have overwritten the untracked local copy. Moving it
  aside for the duration was enough.

**Takeaway.** Verify `git config user.email` against the account's verified
addresses before the first commit. Rewriting is cheap on a solo repo and
expensive on a shared one.

---

## 2. Deployed API returned `{"detail":"Not Found"}` at the root

**Symptom.** The Render URL looked broken in the browser.

**Actual cause.** Nothing was wrong. FastAPI defines `/health`, `/books`,
`/chat`, `/sessions`, … but no handler for `/`, so the bare domain correctly
404s. `/health` returned `{"status":"ok","books_indexed":5}` the whole time.

**Takeaway.** For an API-only service the root path is not a health check.
Probe a real route before concluding a deploy failed.

---

## 3. Vercel scraped backend secrets into the frontend project

**Symptom.** The Vercel import screen pre-filled `LLM_PROVIDER`,
`GROQ_API_KEY`, and `GEMINI_API_KEY` as "3 Detected".

**Actual cause.** Vercel scanned the repo-root `.env.example` — which documents
the *backend* — while the frontend lives in `web/`.

**Fix.** Removed all three; the frontend reads exactly one variable,
`VITE_API_URL`. Keys belong on Render only. Vite inlines only `VITE_`-prefixed
variables, so they would not have leaked into the bundle, but copying a live
secret into a second vendor's dashboard buys nothing.

**The related trap.** Vite inlines `import.meta.env` at **build time**. If
`VITE_API_URL` is missing during the build, the `http://127.0.0.1:8000` fallback
is baked in permanently: the site works on the author's machine and fails for
every visitor, and adding the variable afterwards does nothing until a rebuild.

---

## 4. `ci / docker` failed after 10 seconds

**Symptom.** Backend and frontend CI jobs passed; the docker job failed almost
immediately.

**Why 10s was the clue.** The image build takes minutes — it downloads a ~130 MB
embedding model. Failing in 10s means it never started building.

**Actual cause.** The job used `cache-from`/`cache-to: type=gha`, but GitHub
Actions cache export requires buildx's `docker-container` driver. Without a
`setup-buildx-action` step the runner uses the default `docker` driver, which
cannot export a cache, and the step errors out before the first layer.

**Fix.** Add `- uses: docker/setup-buildx-action@v3` before the build step.

---

## 5. Book search timed out intermittently

**Symptom.** `source lookup failed: HTTPSConnectionPool(host='gutendex.com')
Read timed out. (read timeout=30)` — but only sometimes.

**Measurements.**

| Probe | Time |
|---|---|
| `gutendex.com` from a home connection | 0.32 – 0.75 s |
| `/health` on Render (warm) | 0.41 s |
| `/books` on Render (disk + DB) | 0.40 s |
| `/books/resolve` on Render (calls gutendex) | 17.6 / 24.4 / 26.4 / 40.4 s |

**Actual cause.** Not a cold start, not the frontend, not the app. Gutendex
answers in well under a second from a residential connection but takes 17–40 s
from Render's network — it appears to deprioritise datacenter egress. The client
timeout was 30 s, sitting squarely inside that band, so the call succeeded or
failed essentially at random.

**Fix.** Timeout 30 s → 60 s (`GUTENDEX_TIMEOUT`), one retry on
connection-level errors only, an `lru_cache` on results, and the trailing-slash
URL to skip a 301 redirect on every search. The dialog now says searching can
take up to a minute instead of appearing frozen.

**Honest limit.** This makes searches *reliable*, not *fast*. The ceiling is set
by how gutendex treats that IP range. Removing the dependency from the request
path — caching Gutenberg's catalogue locally — is the only real fix.

**Bonus trap.** When verifying the deploy, repeated identical queries came back
in 0.5 s and looked like proof the new caching was live. It was not: **Gutendex
caches server-side, keyed on the exact query string**, which is why
`TrEaSuRe IsLaNd` still took 24 s while `treasure island` was instant. The local
cache lowercases its key, so if it had been live every case variant would have
hit. An external probe could not distinguish the two.

---

## 6. Render killed the service during indexing (OOM)

**Symptom.** "Web Service LoreLock exceeded its memory limit … triggered an
automatic restart", every time a file was uploaded. Render's free tier caps at
**512 MB**.

**Actual cause.** Two settings written as *speed* optimisations, both fine on a
laptop and fatal in a small container.

**(a) `parallel=1` spawned a second process holding its own copy of the model.**
fastembed branches on `parallel is None`, so any integer — including `1` — takes
the `ParallelWorkerPool` path. Measured live:

```
PARENT  pid=32196   190 MB   <- loaded the model
CHILD   pid=17424   614 MB   <- pool worker, loaded a SECOND copy
                    ------
                    ~804 MB for one embedding job
```

**(b) `batch_size=32` was the dominant term.** Attention intermediates scale with
the batch, and ONNX's arena allocator grows to fit and never returns the pages.
Peak RSS over 128 chunks:

| batch | peak RSS | time |
|---|---:|---|
| 32 (old) | 632 MB | 63.9 s |
| **8 (new)** | **300 MB** | 70.2 s |
| 4 | 246 MB | 74.7 s |

**(c) `threads=os.cpu_count()`.** `os.cpu_count()` does not read the cgroup
quota, so inside a container it reports the *host's* core count (16 on the dev
machine) while a free instance is entitled to a fraction of one core. ONNX sized
its per-thread arenas off that number.

**Fix.** In-process embedding by default (`EMBED_PARALLEL` opts the pool back
in), `EMBED_BATCH=8`, `EMBED_THREADS=2`, all env-tunable and set explicitly in
`render.yaml` so the limit sits next to the plan that imposes it. Vectors are
written straight into a preallocated matrix and normalised in place — the old
path held a list of per-chunk arrays *plus* the finished matrix, then allocated a
third full copy for the division.

**Result.** Peak RSS ~804 MB → **298 MB**, in a single process. 42 tests still
pass.

**Cost.** Indexing is slower: ~10% from the batch change, plus the loss of the
parallel pool. On a fractional-CPU instance that is a real penalty.

---

## 7. Free-tier characteristics that look like bugs

- **Spin-down.** A free instance sleeps after ~15 minutes idle; the next request
  pays container boot + index hydration from Atlas (~30–60 s) and surfaces as a
  *connection failure*, not a 503. The frontend retries `/health` to a 90 s
  deadline rather than reporting the API as unreachable.
- **Ephemeral disk.** No persistent volume on the free plan, which is why
  MongoDB Atlas is the system of record and the local index is rebuilt on boot.
- **Force-push and deploy tracking.** Rewriting history (§1) orphaned the commit
  Render was tracking, so the next deploy had to be triggered manually. Normal
  pushes auto-deploy.

---

## Open items

- **Indexing time on Render is unmeasured.** The corpus averages ~237 words per
  chunk (Shoe Dog: 122,199 words / 510 chunks), and the shipped config embeds at
  ~0.55 s/chunk on a 16-core laptop — ~4.7 min for Shoe Dog. A fractional-CPU
  free instance will be several times slower, but the multiplier has not been
  measured, so no honest estimate exists yet. One timed ingest of a small book
  against the live API would fix that, and everything else scales linearly.
- **Gutenberg catalogue is still fetched per search.** See §5.

---

## Recurring themes

1. **The first-guess cause was wrong in almost every case.** The email, the root
   404, the CI failure, and the OOM all had a plausible wrong explanation that
   would have led to the wrong fix.
2. **Time-to-failure is a diagnostic.** 10 s meant "never started building".
   0.4 s vs 24 s isolated gutendex from the whole rest of the stack.
3. **Laptop-tuned settings are container-hostile.** `os.cpu_count()`, batch
   sizes, and worker pools all assume resources a 512 MB shared instance does
   not have.
4. **Measure before and after.** Every fix here has a number attached; the
   batch-size table is what turned a guess into a decision.
