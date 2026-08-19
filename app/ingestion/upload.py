"""User-supplied books.

Gutenberg only carries public-domain work, so a book still in copyright can
never arrive that way. The legal path for a modern book is the reader's own
DRM-free copy, which is what this module accepts.

It also happens to make the firewall demonstrable rather than merely
plausible: a model that has never seen the text cannot answer from memory,
so a correct reveal is provably a retrieval failure.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.config import RAW

# Formats the parser actually handles. PDF is deliberately absent: extraction
# quality is poor enough that positions become unreliable, and every position
# in this system is an offset into whatever we decide the text is.
ALLOWED_SUFFIXES = {".txt", ".epub", ".html", ".htm", ".xhtml"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_SLUG = re.compile(r"[^a-z0-9]+")


class UploadRejected(ValueError):
    """The file cannot be accepted. The message is shown to the reader."""


def _slug(text: str, limit: int = 60) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")[:limit] or "book"


def _looks_like_epub(data: bytes) -> bool:
    return data[:2] == b"PK" and b"mimetype" in data[:200]


def _decodes_as_text(data: bytes) -> bool:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            data[:4096].decode(enc)
            return True
        except UnicodeDecodeError:
            continue
    return False


def validate(filename: str, data: bytes) -> str:
    """Check the upload and return its normalised suffix.

    Sniffs content rather than trusting the extension -- a .txt that is
    actually a zip would reach the text parser and produce garbage positions.
    """
    if not data:
        raise UploadRejected("That file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"That file is {len(data) / 1e6:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadRejected(
            f"{suffix or 'That file type'} is not supported. "
            f"Upload EPUB, TXT or HTML."
        )

    if suffix == ".epub":
        if not _looks_like_epub(data):
            raise UploadRejected("That does not look like a valid EPUB.")
    elif not _decodes_as_text(data):
        raise UploadRejected(
            "That file is not readable as text. If it is an EPUB, rename it "
            "with a .epub extension."
        )

    return suffix


def epub_metadata(path: Path) -> tuple[str | None, str | None]:
    """Best-effort title/author from EPUB metadata."""
    try:
        from ebooklib import epub
        book = epub.read_epub(str(path))
        title = (book.get_metadata("DC", "title") or [[None]])[0][0]
        author = (book.get_metadata("DC", "creator") or [[None]])[0][0]
        return title, author
    except Exception:  # noqa: BLE001
        return None, None


def store(filename: str, data: bytes) -> tuple[Path, str, str]:
    """Validate and persist an upload. Returns (path, book_id, sha256)."""
    suffix = validate(filename, data)
    digest = hashlib.sha256(data).hexdigest()

    stem = _slug(Path(filename).stem)
    # the digest prefix keeps two different files with the same name apart,
    # while re-uploading the same bytes lands on the same id and cache-hits
    book_id = f"up-{digest[:8]}-{stem}"

    path = RAW / f"{book_id}{suffix}"
    path.write_bytes(data)
    return path, book_id, digest
