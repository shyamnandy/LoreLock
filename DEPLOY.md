# Deployment

Three pieces. The split follows from one constraint: **retrieval is
file-backed and must stay that way.**

The firewall's strongest guarantee is structural — the operand of the
similarity search *is* `E[:n_p]`, so chunks past the reader are absent from
the array rather than filtered out of a result set (see `app/store/vectors.py`).
A remote vector index would express that as `filter: {idx: {$lt: n_p}}`,
demoting a guarantee that cannot fail into a predicate that can. So search
stays local, and MongoDB Atlas holds the *artifacts* instead.

```
   Vercel                    Render (free)                  Atlas (M0)
 ┌────────────┐  HTTPS   ┌──────────────────────┐        ┌──────────────┐
 │  React UI  │ ───────► │  FastAPI             │ ─────► │  books       │
 │  (static)  │          │  sqlite · npy · fts5 │ ◄───── │  payloads    │
 └────────────┘          │  (ephemeral)         │ hydrate└──────────────┘
                         └──────────────────────┘         system of record
```

On boot the container rebuilds its local index from Atlas. On ingest it
writes through. The container's disk is therefore disposable, which is what
allows a free instance with no persistent volume.

## 1. MongoDB Atlas

Create a free **M0** cluster, then:

- **Database Access** → add a user. Save the password.
- **Network Access** → allow `0.0.0.0/0`.

That second step is not laziness: Render's free tier has no static outbound
IPs, so an allowlist cannot work. Compensate with a strong password and a
user scoped to this database only.

Copy the connection string from **Connect → Drivers**.

## 2. Seed Atlas from your machine

Embedding a book takes minutes, and a 512 MB instance is the worst place to
spend them. Ingest locally, then push once:

```bash
# .env
MONGODB_URI=mongodb+srv://user:pass@cluster.xxxxx.mongodb.net/

python scripts/sync_atlas.py status    # what is where
python scripts/sync_atlas.py push      # local -> Atlas
```

Every deploy afterwards hydrates in seconds instead of re-embedding.

## 3. Backend — Render

New → **Blueprint** → point at this repo. It reads `render.yaml`. Then set
three secrets in the dashboard (all `sync: false`, so they never touch git):

| Variable | Value |
|---|---|
| `MONGODB_URI` | the Atlas connection string |
| `GROQ_API_KEY` | your key from console.groq.com/keys |
| `CORS_ORIGINS` | `https://<your-app>.vercel.app` |

Verify:

```bash
curl https://<api-host>/health
# {"status":"ok","books_indexed":5,"answer_provider":"groq"}
```

A non-zero `books_indexed` on a *fresh* container is the proof that
hydration worked — the image ships no books.

## 4. Frontend — Vercel

```bash
cd web
vercel            # or: import the repo at vercel.com/new
```

Set **Root Directory** to `web`, then one environment variable:

| Variable | Value |
|---|---|
| `VITE_API_URL` | `https://<your-api-host>` |

It is `VITE_*` (baked into the bundle at build time) because the browser
calls the API directly — which is also why the backend's CORS list has to
name the Vercel origin. The API already allows any `*.vercel.app` origin by
regex so preview deployments work; set `CORS_ORIGINS` for production anyway.

## 5. Verify the whole loop

1. Open the Vercel URL. Ask a question against a seeded book.
2. Drag the position slider and watch *chunks withheld* fall.
3. Add a book, then hit **Manual Deploy → Restart** in the Render dashboard.
   The book must still be there. That is the test that proves Atlas is doing
   its job, and nothing else tests it.

## What the free tier costs you

**Cold starts.** Free instances sleep after ~15 minutes idle. The next
request pays container boot + hydration + model load — roughly 30–60s. The
UI shows a "waking the server" notice rather than an error (`api.wake`).

**Ingestion needs the instance awake.** Ingest runs for minutes as a
background task, and a free instance stays up only while requests arrive. If
a reader starts an ingest and closes the tab, the job can die with the
instance. It fails cleanly — `reap_orphan_jobs` clears it at the next boot —
but it has to be restarted.

**512 MB of RAM**, which a paid *Starter* instance does **not** increase.
Embedding a long book with onnxruntime resident is the tightest moment in
the system. If it OOMs, the answer is a Standard instance, not Starter.

To remove the cold starts, change `plan: free` to a paid plan in
`render.yaml`. No code change is needed, and with Atlas as the system of
record you still do not need a disk.

## Running locally

```bash
# terminal 1 — API
.venv/Scripts/python.exe -m uvicorn app.api:api --port 8000

# terminal 2 — UI
cd web && npm run dev        # http://localhost:3000
```

`web/.env.local` should contain `VITE_API_URL=http://127.0.0.1:8000`.

`MONGODB_URI` is optional locally. Unset, `app/store/remote.py` degrades to a
no-op: nothing is pushed, nothing is hydrated, and the app behaves exactly as
it did before Atlas existed. The test suite runs this way on purpose.
