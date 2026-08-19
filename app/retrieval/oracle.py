"""The spoiler oracle -- STORE B, air-gapped.

To tell "this is answered on page 300" apart from "this is never in the
book" you need something that can see the future. That capability is
confined to this module, and it is deliberately NOT an LLM call: the only
value that crosses back is a boolean.

No book text read here is ever returned, logged into a prompt, or handed to
a model. If you change that, you have removed the air gap.
"""
from __future__ import annotations

import numpy as np

from app.config import ORACLE_THRESHOLD
from app.store import vectors


def answered_later(book_id: str, n_p: int, question: str,
                   threshold: float = ORACLE_THRESHOLD) -> bool:
    """True when material past the reader position looks like it answers the
    question substantially better than anything the reader has already seen.
    """
    full = vectors.load(book_id)
    if n_p >= len(full):
        return False

    q = vectors.embed_query(question)
    future_best = float(np.max(full[n_p:] @ q))
    past_best = float(np.max(full[:n_p] @ q)) if n_p else -1.0

    return future_best >= threshold and future_best > past_best + 0.05
