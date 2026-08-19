"""Reciprocal Rank Fusion.

Two retrieval paths produce incomparable scores -- cosine similarity in
[-1,1] and BM25 in an unbounded negative range. RRF discards the magnitudes
and fuses on rank alone, which is why it needs no normalisation and no
per-corpus tuning.
"""
from __future__ import annotations

from app.config import RRF_K


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Fuse ranked id lists into {id: fused_score}, best first when sorted."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def fuse(vector_ids: list[str], keyword_ids: list[str], k: int = RRF_K
         ) -> list[tuple[str, float, str]]:
    """Returns [(chunk_id, fused_score, via)] sorted best first."""
    scores = rrf([vector_ids, keyword_ids], k)
    vset, kset = set(vector_ids), set(keyword_ids)

    def via(cid: str) -> str:
        if cid in vset and cid in kset:
            return "both"
        return "vector" if cid in vset else "keyword"

    return sorted(
        ((cid, score, via(cid)) for cid, score in scores.items()),
        key=lambda t: t[1], reverse=True,
    )
