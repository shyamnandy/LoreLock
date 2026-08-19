"""Format adapters: raw bytes in, plain text out.

Plain text, deliberately -- not Markdown. Every position in this system is a
word offset into whatever this module decides the text is, so any syntax we
introduce (`**`, `#`, `_`) would become words in the stream, shift every
offset after it, and pollute the embeddings. Markup is discarded, not
translated.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from bs4 import BeautifulSoup

try:                                        # bs4 >= 4.11
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:                         # pragma: no cover
    pass

# Tags that end a line of prose. Everything else -- <em>, <strong>, <a>,
# <span> -- is inline and must NOT introduce whitespace, or "the <em>light</em>
# for" becomes "the \nlight\n for" and paragraph structure is destroyed.
_BLOCK = (
    "p", "div", "br", "hr", "li", "tr", "blockquote", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
)


def from_txt(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def from_html(path_or_markup: Path | str) -> str:
    markup = (path_or_markup.read_text(encoding="utf-8", errors="replace")
              if isinstance(path_or_markup, Path) else path_or_markup)
    soup = BeautifulSoup(markup, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Mark block boundaries explicitly, then extract with no separator, so
    # inline emphasis stays welded to the words around it.
    for tag in soup.find_all(_BLOCK):
        tag.insert_before("\n")
        tag.insert_after("\n")

    return soup.get_text("")


def from_epub(path: Path) -> str:
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path))

    # Follow the spine, which is the reading order the publisher declared.
    # Iterating get_items() returns files in archive order, which is not
    # necessarily narrative order -- and narrative order is the one thing
    # this whole system depends on.
    by_id = {item.get_id(): item for item in book.get_items()}
    ordered = [by_id[idref] for idref, _linear in book.spine if idref in by_id]
    if not ordered:
        ordered = [i for i in book.get_items()
                   if i.get_type() == ebooklib.ITEM_DOCUMENT]

    parts = [
        from_html(item.get_content().decode("utf-8", "replace"))
        for item in ordered
        if item.get_type() == ebooklib.ITEM_DOCUMENT
    ]
    return "\n\n".join(parts)


def parse(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return from_epub(path)
    if suffix in {".html", ".htm", ".xhtml"}:
        return from_html(path)
    return from_txt(path)
