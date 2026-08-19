"""Embedding matrix, stored as one row-per-chunk array in narrative order.

This is where the firewall stops being a policy and becomes a fact about
memory layout. Because row `i` is chunk `idx == i`, the set of chunks a
reader may see is the slice `E[:n_p]` -- a contiguous prefix. A similarity
search over that slice cannot return chunk `n_p` or beyond, because those
rows are not in the operand.

There is no filter here to forget.
"""
from __future__ import annotations

import functools
import os
import threading
import time

import numpy as np

from app.config import EMBED_DIM, EMBED_MODEL, VEC_DIR

_model_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def _embedder():
    import os

    from fastembed import TextEmbedding
    # parallel=1 below: fastembed's worker-pool path is pathological on
    # Windows (multi-GB RSS, no throughput gain). One process, many threads.
    return TextEmbedding(model_name=EMBED_MODEL, threads=os.cpu_count())


def embed_texts(texts: list[str], progress: bool = False) -> np.ndarray:
    """Embed and L2-normalise, so cosine similarity is a plain dot product."""
    with _model_lock:
        model = _embedder()
        out: list[np.ndarray] = []
        for i, vec in enumerate(model.embed(texts, batch_size=32, parallel=1)):
            out.append(vec)
            if progress and (i + 1) % 64 == 0:
                print(f"  embedded {i + 1}/{len(texts)}", flush=True)
        vectors = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def _path(book_id: str):
    return VEC_DIR / f"{book_id}.npy"


def save(book_id: str, matrix: np.ndarray) -> None:
    """Write the matrix atomically.

    Writing in place risks leaving a truncated .npy behind if the process
    dies mid-write, and the chunk table would then point at rows that do not
    exist. Write beside it, then swap.
    """
    if matrix.ndim != 2 or matrix.shape[1] != EMBED_DIM:
        raise ValueError(f"expected (n,{EMBED_DIM}) matrix, got {matrix.shape}")

    final = _path(book_id)
    # the temp name must itself end in .npy -- np.save silently appends the
    # extension otherwise, and the replace would then find nothing to move
    tmp = final.with_suffix(".tmp.npy")
    np.save(tmp, matrix.astype(np.float32))
    load.cache_clear()          # drop our own handle before replacing

    # os.replace is atomic on POSIX and Windows alike, but on Windows it
    # fails outright if anything else holds the target open even for an
    # instant -- OneDrive sync and antivirus scanners both do. Retry briefly
    # rather than losing a five-minute embedding run to a transient lock.
    last: OSError | None = None
    for attempt in range(6):
        try:
            os.replace(tmp, final)
            return
        except PermissionError as exc:
            last = exc
            time.sleep(0.5 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    raise OSError(
        f"could not replace {final.name} after 6 attempts -- another process "
        f"is holding it open (OneDrive sync? a running API server?)"
    ) from last


def exists(book_id: str) -> bool:
    return _path(book_id).exists()


@functools.lru_cache(maxsize=4)
def load(book_id: str) -> np.ndarray:
    """Full-book matrix. STORE B -- only `app.retrieval.oracle` may use this
    directly, and only to produce a boolean.

    Read fully into memory rather than memory-mapped. These matrices are
    ~1-3 MB, so mmap buys nothing, and on Windows an open mapping makes the
    file unwritable: a re-index would fail with EINVAL while the API server
    held the old one open.
    """
    return np.load(_path(book_id))


def prefix(book_id: str, n_p: int) -> np.ndarray:
    """STORE A: the session's searchable matrix.

    The returned array has exactly `n_p` rows. Nothing downstream can widen
    it, because the rows past the boundary are simply absent.
    """
    full = load(book_id)
    if n_p < 0:
        raise ValueError("n_p must be non-negative")
    if n_p > len(full):
        # n_p is a COUNT of rows in `chunks`, so it can only exceed the matrix
        # if the two have drifted -- a book re-chunked without re-embedding.
        # Slicing would silently clamp and quietly return the whole book.
        raise ValueError(
            f"n_p={n_p} exceeds the {len(full)}-row matrix for {book_id!r}; "
            f"the chunk table and embeddings have drifted -- re-ingest"
        )
    return full[:n_p]
