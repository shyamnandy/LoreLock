"""Embedding backend: fastembed / ONNX, in this process.

One backend, no provider switch. The model is ~190 MB resident, which is the
memory high-water mark of the whole app -- see CHALLENGES.md §6 for the
measurements and for why `EMBED_BATCH`, `EMBED_THREADS` and `EMBED_PARALLEL`
are tunable at all.

Embedding is not only an ingest-time cost: `app.retrieval.firewall` embeds
every incoming question through `vectors.embed_query`, so the model is
resident for as long as the API serves.

Returns raw (n, dim) float32. Normalisation belongs to the caller in
`app.store.vectors`, so it happens exactly once.
"""
from __future__ import annotations

import os
import threading

import numpy as np

import app.config


class EmbedError(RuntimeError):
    """Embedding could not be produced. Never raised for an empty input."""


# Serialises access to the ONNX session, which is not thread-safe.
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


def model_name() -> str:
    return app.config.EMBED_MODEL


def embed(texts: list[str], *, input_type: str = "document",
          progress: bool = False) -> np.ndarray:
    """Raw, un-normalised (len(texts), EMBED_DIM) float32.

    `input_type` is accepted and ignored -- bge draws no query/document
    distinction. It stays in the signature because `vectors.embed_query`
    passes it and the asymmetry is real for other model families.
    """
    if not texts:
        return np.empty((0, app.config.EMBED_DIM), dtype=np.float32)

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
