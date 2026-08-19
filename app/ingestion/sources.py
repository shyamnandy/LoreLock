"""Book acquisition from permitted sources.

MVP policy: Project Gutenberg (public domain) via the Gutendex API, plus
local user upload. No scraping of arbitrary copyrighted-book sites.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import requests

from app.config import RAW

GUTENDEX = "https://gutendex.com/books"
UA = {"User-Agent": "spoiler-safe-reading-companion/0.1 (educational project)"}

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


def search(query: str, limit: int = 5) -> list[SourceCandidate]:
    """Search Gutendex for a title/author string."""
    resp = requests.get(GUTENDEX, params={"search": query}, headers=UA, timeout=30)
    resp.raise_for_status()
    out: list[SourceCandidate] = []

    for item in resp.json().get("results", [])[:limit]:
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
    return out


def download(candidate: SourceCandidate) -> tuple[object, str]:
    """Fetch the permitted file. Returns (path, sha256)."""
    path = RAW / f"{candidate.book_id}.{candidate.fmt}"
    if path.exists():
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    resp = requests.get(candidate.url, headers=UA, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path, hashlib.sha256(resp.content).hexdigest()
