# Backend image.
#
# The API needs a real filesystem: SQLite, the FTS5 index and the embedding
# matrices are files, and retrieval slices those matrices directly. There is
# no remote copy -- whatever is under DATA_DIR IS the library, so that path
# must be durable if the index is to survive a redeploy.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Baked into the image, NOT into DATA_DIR: the weights are the same for
    # every deploy, so paying to re-download them on each cold start is pure
    # latency. This path is populated at build time, below.
    FASTEMBED_CACHE_PATH=/opt/fastembed

WORKDIR /srv

# ca-certificates is required for TLS to the model host and the LLM provider
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Created before anything populates a path it will later need to open.
RUN useradd --create-home --uid 10001 companion

# Pre-download the embedding model into the image. Without this the first
# request after every cold start pays a ~130 MB download before it can
# answer, and on a host with no persistent disk that is *every* cold start.
# Owned by the runtime user: fastembed mkdir -p's its cache on every load,
# and that must not fail just because root did the downloading.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
 && chown -R companion:companion /opt/fastembed

COPY app ./app
COPY scripts ./scripts

# The prebuilt, public-domain-only library. Read-only and outside DATA_DIR:
# the container copies it into /data on first boot (app.store.seed), so a
# host with no persistent disk still comes up with books instead of an empty
# shelf. Rebuild it with scripts/build_seed.py and commit the result --
# Render builds from the git checkout, so an untracked seed reaches nothing.
COPY seed ./seed
ENV SEED_DIR=/srv/seed

# Writable state. Deliberately NOT a VOLUME: a declared volume shadows the
# path, and Render mounts its own disk here anyway if a blueprint asks for
# one. NOTE: without a mounted disk this is ephemeral. The seed above is
# what makes that survivable -- but only for the books baked in. Anything
# ingested at runtime still dies with the container.
ENV DATA_DIR=/data
RUN mkdir -p /data && chown -R companion:companion /data /srv
USER companion

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1

# Shell form so ${PORT} expands -- Render assigns the port at runtime -- but
# `exec` so uvicorn REPLACES the shell and becomes PID 1. Without it the
# platform's SIGTERM goes to /bin/sh, which does not forward it, and every
# redeploy ends in a hard kill instead of a graceful shutdown.
#
# One worker, not several: vectors.load caches matrices per process and
# invalidates only its own cache, so a second worker would serve a stale
# matrix after a re-ingest.
CMD exec uvicorn app.api:api --host 0.0.0.0 --port ${PORT:-8000} --workers 1
