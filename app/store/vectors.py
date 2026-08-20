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


# Memory, not speed, is the binding constraint on a small container. The
# ONNX model is ~190 MB resident, and both knobs below used to multiply
# that: `parallel=1` spawned a worker PROCESS that loaded its own second
# copy, and `threads=os.cpu_count()` sized ONNX's per-thread arenas off the
# HOST's core count -- os.cpu_count() does not read the cgroup quota, so a
# container entitled to a fraction of a core still saw 16. Measured on one
# 512-chunk job: 190 MB parent + 614 MB worker = ~800 MB, an instant OOM
# kill on a 512 MB instance. Both are now opt-in for hosts with headroom.
_PARALLEL = int(os.environ["EMBED_PARALLEL"]) if os.environ.get("EMBED_PARALLEL") else None
_THREADS = int(os.environ.get("EMBED_THREADS", "2"))

# The single biggest lever on peak memory. Attention intermediates scale
# with the batch, so ONNX's arena grows with it and never gives the pages
# back. Measured over 128 chunks: batch 32 peaked at 632 MB, batch 8 at
# 300 MB for ~10% more wall time, batch 4 at 246 MB. 8 is the default
# because it fits a 512 MB instance with room for the rest of the app;
# raise it on a bigger box, drop it to 4 if an ingest still gets killed.
_BATCH = int(os.environ.get("EMBED_BATCH", "8"))


@functools.lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL, threads=_THREADS)


def embed_texts(texts: list[str], progress: bool = False) -> np.ndarray:
    """Embed and L2-normalise, so cosine similarity is a plain dot product."""
    with _model_lock:
        model = _embedder()
        # Written straight into the destination matrix. Accumulating a list
        # of per-chunk arrays and calling np.array on it holds both the list
        # and the finished matrix at once, for no benefit.
        #
        # fastembed branches on `parallel is None`, so any int -- including
        # 1 -- takes the ParallelWorkerPool path and pays for a second model.
        # A caller that sets EMBED_PARALLEL outside a server needs an
        # `if __name__ == "__main__"` guard, or the spawned child re-imports
        # the module and dies.
        n = 0
        matrix = np.empty((len(texts), EMBED_DIM), dtype=np.float32)
        for i, vec in enumerate(model.embed(texts, batch_size=_BATCH,
                                            parallel=_PARALLEL)):
            matrix[i] = vec
            n = i + 1
            if progress and n % 64 == 0:
                print(f"  embedded {n}/{len(texts)}", flush=True)
        if n != len(texts):
            raise RuntimeError(
                f"embedder returned {n} vectors for {len(texts)} chunks")
    # In place: `matrix / norms` would allocate a second full matrix, and
    # for a long book that copy is the difference between fitting and not.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    matrix /= norms
    return matrix


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


def delete(book_id: str) -> bool:
    """Drop a book's matrix. Returns False if there was nothing to drop.

    The lru_cache must be cleared before the unlink, not after: on Windows a
    cached handle is enough to make the file undeletable, and a matrix left
    on disk under a reused book id would be silently searched by the next
    ingest of the same title.
    """
    load.cache_clear()
    final = _path(book_id)
    gone = final.exists()
    final.unlink(missing_ok=True)
    final.with_suffix(".tmp.npy").unlink(missing_ok=True)   # a crashed save
    return gone


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
