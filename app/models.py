"""Core domain types.

The single most important invariant in this codebase:

    chunks are stored in narrative order, so `idx` is monotonic in both
    `word_start` and `word_end`.

Everything the firewall does rests on that. It is asserted at ingest time
(see app.store.db.insert_chunks) rather than trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import WORDS_PER_PAGE

Intent = Literal[
    "character", "event", "motivation",
    "relationship", "summary", "clue", "prediction",
]

# "error" is deliberately distinct from the two refusals: a provider failure
# that presents as "not enough evidence" hides an outage behind correct-looking
# spoiler-safe behaviour, and an eval will happily score it as success.
Outcome = Literal["answered", "refused_spoiler", "refused_thin", "error"]


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    book_id: str
    idx: int
    text: str
    chapter: int
    chapter_title: str
    word_start: int
    word_end: int

    @property
    def page(self) -> int:
        """Synthetic page number. The UI speaks pages; the firewall does not."""
        return self.word_start // WORDS_PER_PAGE + 1


@dataclass(frozen=True, slots=True)
class Book:
    id: str
    title: str
    author: str
    source: str
    source_url: str
    license: str
    total_words: int
    n_chunks: int
    # The reader's own edition, when they told us: a printed book's page count
    # bears no relation to total_words // 250, and the slider has to speak the
    # number that is printed in the corner of the page in front of them.
    edition_pages: int | None = None

    @property
    def words_per_page(self) -> int:
        """Words the slider advances per page.

        With an edition declared this is derived, so that page N of the
        reader's edition maps to the same *fraction* of the text: page 30 of
        100 lands at 30% of the words, whatever our own paging would call it.
        """
        if self.edition_pages:
            return max(1, round(self.total_words / self.edition_pages))
        return WORDS_PER_PAGE

    @property
    def total_pages(self) -> int:
        if self.edition_pages:
            return self.edition_pages
        return max(1, self.total_words // WORDS_PER_PAGE)


@dataclass(frozen=True, slots=True)
class Evidence:
    """A retrieved chunk plus why it was retrieved. Carries its own position
    so downstream code can re-assert the bound without a database round-trip."""
    chunk: Chunk
    score: float        # RRF fused score -- encodes RANK ONLY, not relevance
    via: str            # "vector" | "keyword" | "both"
    similarity: float   # raw cosine against the query; the relevance signal


class FirewallBreach(RuntimeError):
    """Raised when a chunk past the reader position reaches the answer path.

    This is deliberately fatal. A silent filter here would conceal the bug
    that produced it; the whole value of the system is that this never fires.
    """
