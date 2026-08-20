"""Book acquisition from permitted sources.

MVP policy: Project Gutenberg (public domain) via the Gutendex API, plus
local user upload. No scraping of arbitrary copyrighted-book sites.
"""
from __future__ import annotations

import functools
import hashlib
import os
import re
import time
from dataclasses import dataclass

import requests

from app.config import RAW

# Trailing slash deliberately: the bare path 301s here, and requests follows
# the redirect, paying an extra round-trip on every single search.
GUTENDEX = "https://gutendex.com/books/"
UA = {"User-Agent": "spoiler-safe-reading-companion/0.1 (educational project)"}

# Gutendex answers in well under a second from a home connection but
# routinely takes 20-40s from a datacenter IP -- it appears to deprioritise
# cloud egress. A deployed backend therefore needs a far more generous
# budget than a laptop does, and the old 30s sat squarely inside that band,
# so lookups failed intermittently rather than cleanly. Env-tunable so the
# budget can be adjusted on the host without a redeploy.
_TIMEOUT = float(os.environ.get("GUTENDEX_TIMEOUT", "60"))
_RETRIES = int(os.environ.get("GUTENDEX_RETRIES", "1"))

_TEXT_KEYS = (
    "text/plain; charset=utf-8",
    "text/plain; charset=us-ascii",
    "text/plain",
)


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    book_id: str
    title: str
    author: str
    url: str
    fmt: str
    source: str = "gutenberg"
    license: str = "public domain (Project Gutenberg)"
    download_count: int = 0


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def _fetch(query: str) -> list[dict]:
    """GET the catalogue, retrying a timeout rather than surfacing it.

    Only connection-level failures are retried. An HTTP error is a real
    answer from the service and repeating it would just double the wait.
    """
    last: Exception | None = None
    for attempt in range(_RETRIES + 1):
        try:
            resp = requests.get(GUTENDEX, params={"search": query},
                                headers=UA, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
            if attempt < _RETRIES:
                time.sleep(2)
    raise last  # type: ignore[misc]


@functools.lru_cache(maxsize=256)
def _search_cached(query: str, limit: int) -> tuple[SourceCandidate, ...]:
    """Cached because the Gutenberg catalogue is effectively static, while
    each miss costs the full datacenter-latency penalty above. lru_cache
    does not memoise exceptions, so a failed lookup is retried next time
    rather than being pinned for the life of the process.
    """
    out: list[SourceCandidate] = []

    for item in _fetch(query)[:limit]:
        formats = item.get("formats", {})
        url = fmt = None
        for key in _TEXT_KEYS:
            if key in formats and not formats[key].endswith(".zip"):
                url, fmt = formats[key], "txt"
                break
        if url is None:
            for key, value in formats.items():
                if key.startswith("application/epub"):
                    url, fmt = value, "epub"
                    break
        if url is None:
            continue

        authors = item.get("authors") or [{"name": "Unknown"}]
        title = item.get("title", "Untitled")
        out.append(SourceCandidate(
            book_id=f"gb{item['id']}-{_slug(title)}",
            title=title,
            author=authors[0].get("name", "Unknown"),
            url=url,
            fmt=fmt,
            download_count=item.get("download_count", 0),
        ))
    return tuple(out)


def search(query: str, limit: int = 5) -> list[SourceCandidate]:
    """Search Gutendex for a title/author string."""
    # Normalised for the cache key; Gutendex matching is case-insensitive,
    # so this only widens the hit rate, it does not change the results.
    return list(_search_cached(query.strip().lower(), limit))


def download(candidate: SourceCandidate) -> tuple[object, str]:
    """Fetch the permitted file. Returns (path, sha256)."""
    path = RAW / f"{candidate.book_id}.{candidate.fmt}"
    if path.exists():
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    resp = requests.get(candidate.url, headers=UA, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path, hashlib.sha256(resp.content).hexdigest()
