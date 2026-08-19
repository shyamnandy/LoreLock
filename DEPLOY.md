# Deployment

Two services. The split is not incidental: the backend needs a **persistent
disk** (SQLite, the FTS5 index and the `.npy` embedding matrices are files,
and ingestion writes to them for minutes at a time), which serverless
platforms do not provide. Vercel serves the UI it is good at; the stateful
half lives on a host with a volume.

```
   Vercel                          Render / Fly
 ┌──────────────┐   HTTPS/CORS   ┌──────────────────────┐
 │  Next.js UI  │ ─────────────► │  FastAPI  +  volume  │
 │  (static)    │                │  sqlite · npy · fts5 │
 └──────────────┘                └──────────────────────┘
```

## 1. Backend

The repo ships a `Dockerfile` and a `render.yaml` blueprint.

**Render** — New → Blueprint → point at this repo. It reads `render.yaml`,
builds the image and mounts a 1 GB disk at `/data`. Then set two secrets in
the dashboard (they are `sync: false`, so they never touch git):

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | your key from console.groq.com/keys |
| `CORS_ORIGINS` | `https://<your-app>.vercel.app` |

`DATA_DIR=/data` is already set by the blueprint, which is what puts the
index on the mounted disk instead of the ephemeral container filesystem.

**Fly.io** — `fly launch --no-deploy`, then `fly volumes create companion_data
--size 1`, mount it at `/data` in `fly.toml`, `fly secrets set GROQ_API_KEY=…
CORS_ORIGINS=…`, and `fly deploy`.

### Seeding the index

The disk starts empty, so no books are indexed. Either:

```bash
curl -X POST https://<api-host>/books/ingest \
     -H 'Content-Type: application/json' \
     -d '{"query": "The Adventures of Sherlock Holmes"}'
```

(expect several minutes — it downloads, chunks and embeds), or run
`scripts/ingest.py` locally and copy `data/db/` onto the volume.

> Render's free tier sleeps after inactivity and **its disk is wiped on
> instance replacement**. For a demo that must stay warm, use a paid instance
> or re-seed on boot.

## 2. Frontend

```bash
cd web
vercel            # or: import the repo at vercel.com/new
```

Set **Root Directory** to `web` in the Vercel project settings, then add one
environment variable:

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://<your-api-host>` |

It is `NEXT_PUBLIC_*` because the browser calls the API directly — which is
also why the backend's CORS list has to name the Vercel origin. The API
already allows any `*.vercel.app` origin by regex so preview deployments
work, but set `CORS_ORIGINS` to your real domain for production.

## 3. Verify

```bash
curl https://<api-host>/health
# {"status":"ok","books_indexed":1,"answer_provider":"groq"}
```

Then open the Vercel URL, drag the slider, and watch *chunks withheld* fall.

## Running both locally

```bash
# terminal 1 — API
.venv/Scripts/python.exe -m uvicorn app.api:api --port 8000

# terminal 2 — UI
cd web && npm run dev        # http://localhost:3000
```

`web/.env.local` should contain `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`.

## Why not put the whole thing on Vercel?

It is possible — the data is only ~4 MB, so the SQLite file and the `.npy`
matrices fit inside a function bundle as read-only build artifacts, and the
NumPy prefix slice would work untouched. What you lose is runtime ingestion
(books would be added locally and redeployed) and server-side sessions.
That variant is a reasonable follow-up if you want a single-platform deploy;
the trade is state for simplicity.
