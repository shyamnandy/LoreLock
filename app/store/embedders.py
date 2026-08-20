"""Embedding backends.

Select with EMBED_PROVIDER: local | voyage

The shape of this module deliberately mirrors `app.agents.llm`: a dict of
backends, one env var to choose, an explicit error for an unknown name.

Why a remote option exists at all: the local backend loads a ~190 MB ONNX
model into the API process, and on a 512 MB instance that is most of the
budget before a single request arrives. Moving *generation* off-box removes
that, while retrieval stays exactly where it is -- `vectors.prefix()` still
slices `E[:n_p]`, so the firewall keeps its structural bound. See
CHALLENGES.md for the measurements, and app/retrieval/firewall.py for why
that bound is not negotiable.

Backends return raw (n, dim) float32. Normalisation belongs to the caller in
`app.store.vectors`, so it happens exactly once regardless of provider.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

import app.config


class EmbedError(RuntimeError):
    """Embedding could not be produced. Never raised for an empty input."""


# -- local (fastembed / ONNX) ---------------------------------------------

# Serialises access to the ONNX session, which is not thread-safe. Only the
# local backend needs this; holding it across a network call would pointlessly
# serialise the remote one.
_model_lock = threading.Lock()

# Memory, not speed, is the binding constraint on a small container. See
# CHALLENGES.md §6: `parallel=1` spawns a worker process holding a second
# copy of the model, and batch size drives ONNX's arena allocator.
_PARALLEL = int(os.environ["EMBED_PARALLEL"]) if os.environ.get("EMBED_PARALLEL") else None
_THREADS = int(os.environ.get("EMBED_THREADS", "2"))
_BATCH = int(os.environ.get("EMBED_BATCH", "8"))


def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=app.config.EMBED_MODEL, threads=_THREADS)


_cached_model = None


def _local(texts: list[str], input_type: str, progress: bool) -> np.ndarray:
    """In-process ONNX. `input_type` is ignored -- bge makes no distinction."""
    global _cached_model
    with _model_lock:
        if _cached_model is None:
            _cached_model = _embedder()
        # Written straight into the destination matrix. Accumulating a list of
        # per-chunk arrays and calling np.array on it holds both the list and
        # the finished matrix at once, for no benefit.
        n = 0
        matrix = np.empty((len(texts), app.config.EMBED_DIM), dtype=np.float32)
        for i, vec in enumerate(_cached_model.embed(texts, batch_size=_BATCH,
                                                    parallel=_PARALLEL)):
            matrix[i] = vec
            n = i + 1
            if progress and n % 64 == 0:
                print(f"  embedded {n}/{len(texts)}", flush=True)
        if n != len(texts):
            raise EmbedError(
                f"embedder returned {n} vectors for {len(texts)} chunks")
    return matrix


# -- voyage (MongoDB's embedding API) -------------------------------------

# Voyage is reachable two ways and the keys are NOT interchangeable: a
# native key from voyageai.com works against api.voyageai.com, while a key
# minted in the Atlas console only works against MongoDB's own endpoint.
# Using the wrong pair returns a 403 saying exactly that. Overridable so
# either route works without a code change.
_VOYAGE_URL = os.environ.get("VOYAGE_URL",
                             "https://api.voyageai.com/v1/embeddings")

# Sized against the TOKEN limit, not the item limit. A free-tier project is
# 10k tokens/min and 3 requests/min; at ~315 tokens per chunk a 24-item batch
# is ~7.5k tokens, so one request fits inside the per-minute budget with room
# to spare. Raising this past ~32 makes every request exceed the whole budget
# and 429 forever -- raise the project's limit first, then this.
_VOYAGE_BATCH = int(os.environ.get("VOYAGE_BATCH", "24"))
_VOYAGE_TIMEOUT = float(os.environ.get("VOYAGE_TIMEOUT", "60"))
# Rate limits are the expected case here, not an anomaly: a full book is many
# minutes of budget, so a 429 has to be waited out rather than given up on.
_VOYAGE_RETRIES = int(os.environ.get("VOYAGE_RETRIES", "6"))
_VOYAGE_BACKOFF = float(os.environ.get("VOYAGE_BACKOFF", "20"))


def _voyage_post(batch: list[str], input_type: str) -> list[list[float]]:
    import requests

    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise EmbedError("VOYAGE_API_KEY is not set")

    payload = {
        "input": batch,
        "model": app.config.EMBED_MODEL,
        # Voyage embeds queries and documents into the same space but tunes
        # for the asymmetry. Getting this backwards quietly costs recall.
        "input_type": input_type,
        "output_dimension": app.config.EMBED_DIM,
    }
    headers = {"Authorization": f"Bearer {key}"}

    last: Exception | None = None
    for attempt in range(_VOYAGE_RETRIES + 1):
        try:
            resp = requests.post(_VOYAGE_URL, json=payload, headers=headers,
                                 timeout=_VOYAGE_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = EmbedError(f"cannot reach voyage: {exc}")
        else:
            if resp.status_code == 200:
                try:
                    data = resp.json()["data"]
                except (KeyError, ValueError) as exc:
                    last = EmbedError(
                        f"unexpected voyage response: {resp.text[:300]}")
                else:
                    # The API documents ordered results, but the index is
                    # present and cheap to honour -- an out-of-order batch
                    # would silently pair the wrong vector with the wrong
                    # chunk, which retrieval would never surface as an error.
                    return [row["embedding"]
                            for row in sorted(data, key=lambda r: r["index"])]
            elif resp.status_code == 429 or resp.status_code >= 500:
                last = EmbedError(f"voyage {resp.status_code}: {resp.text[:200]}")
                # The server knows how long it wants; prefer its answer to
                # our guess. Absent the header, wait a whole rate-limit
                # window rather than the few seconds a 5xx would want.
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if (retry_after or "").replace(".", "", 1).isdigit() else _VOYAGE_BACKOFF
                    if attempt < _VOYAGE_RETRIES:
                        time.sleep(wait)
                        continue
            else:
                # Anything else is a real answer about the request itself --
                # a bad key, an unknown model, a malformed body. It will be
                # just as wrong on the third attempt, so fail now.
                raise EmbedError(f"voyage {resp.status_code}: {resp.text[:300]}")
        if attempt < _VOYAGE_RETRIES:
            time.sleep(2 * (attempt + 1))
    raise EmbedError(str(last))


def _voyage(texts: list[str], input_type: str, progress: bool) -> np.ndarray:
    matrix = np.empty((len(texts), app.config.EMBED_DIM), dtype=np.float32)
    for start in range(0, len(texts), _VOYAGE_BATCH):
        batch = texts[start:start + _VOYAGE_BATCH]
        rows = _voyage_post(batch, input_type)
        if len(rows) != len(batch):
            raise EmbedError(
                f"voyage returned {len(rows)} vectors for {len(batch)} inputs")
        got = len(rows[0])
        if got != app.config.EMBED_DIM:
            raise EmbedError(
                f"voyage returned {got}-dim vectors but EMBED_DIM is "
                f"{app.config.EMBED_DIM}. Set VOYAGE_DIM={got} (and re-index every "
                f"book -- stored matrices are the old width)."
            )
        matrix[start:start + len(batch)] = np.asarray(rows, dtype=np.float32)
        if progress:
            print(f"  embedded {min(start + len(batch), len(texts))}/"
                  f"{len(texts)}", flush=True)
    return matrix


_BACKENDS = {"local": _local, "voyage": _voyage}


def provider_name() -> str:
    return app.config.EMBED_PROVIDER


def available() -> bool:
    """True if the configured backend could actually run right now."""
    if provider_name() == "voyage":
        return bool(os.environ.get("VOYAGE_API_KEY"))
    return True


def embed(texts: list[str], *, input_type: str = "document",
          progress: bool = False) -> np.ndarray:
    """Raw, un-normalised (len(texts), EMBED_DIM) float32."""
    if not texts:
        return np.empty((0, app.config.EMBED_DIM), dtype=np.float32)
    name = provider_name()
    backend = _BACKENDS.get(name)
    if backend is None:
        raise EmbedError(
            f"unknown EMBED_PROVIDER {name!r}; expected one of {sorted(_BACKENDS)}")
    return backend(texts, input_type, progress)
