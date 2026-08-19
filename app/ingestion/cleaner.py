"""Strip source boilerplate and recover chapter structure from raw text.

Clean extraction is the least glamorous and most consequential part of the
pipeline: every position in the system is an offset into whatever this
module decides the book's text is.
"""
from __future__ import annotations

import re

from app.ingestion.chunker import Section

_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                    re.IGNORECASE | re.DOTALL)
_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                  re.IGNORECASE | re.DOTALL)

# a chapter heading on its own line: "CHAPTER IV", "Chapter 12.", "III.",
# "ADVENTURE I. A SCANDAL IN BOHEMIA"
_HEADING = re.compile(
    r"^[ \t]*("
    r"(?:CHAPTER|Chapter|BOOK|Book|PART|Part|ADVENTURE|Adventure|LETTER|Letter)"
    r"\s+[0-9IVXLCDM]+[.:]?.*"
    r"|[IVXLCDM]{1,7}[.]\s*.{0,60}"
    r")[ \t]*$"
)

_RULE = re.compile(r"^[ \t]*[-_*=~]{3,}[ \t]*$", re.MULTILINE)

# Transcription artefacts, not prose. These pollute the embedding -- an
# "[Illustration]" token carries no narrative meaning -- and they surface
# verbatim in quoted evidence, which is how they reached the answers.
_ILLUSTRATION = re.compile(
    r"\[\s*(?:Illustration|Frontispiece|Plate)[^\]]*\]", re.IGNORECASE
)
_FOOTNOTE_MARK = re.compile(r"\[\d{1,3}\]")
# _word_ is Gutenberg's convention for italics: keep the word, drop the marks
_ITALIC = re.compile(r"(?<![A-Za-z0-9])_([^_\n]{1,80})_(?![A-Za-z0-9])")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the licence headers/footers that bracket every Gutenberg text."""
    if m := _START.search(text):
        text = text[m.end():]
    if m := _END.search(text):
        text = text[:m.start()]
    # the produced-by / transcriber block that often follows the start marker
    text = re.sub(r"^\s*(Produced by|E-?text prepared by|Transcribed from).*?\n\n",
                  "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = _ILLUSTRATION.sub("", text)
    text = _FOOTNOTE_MARK.sub("", text)
    text = _ITALIC.sub(r"\1", text)
    text = _RULE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_sections(text: str) -> list[Section]:
    """Split cleaned text on chapter headings.

    Falls back to a single section when no headings are found -- positions
    still work, only the chapter metadata is coarser.
    """
    lines = text.split("\n")
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if not line.strip() or len(line.strip()) > 90:
            continue
        if _HEADING.match(line):
            # require blank space around it, so mid-paragraph roman numerals
            # and dialogue are not mistaken for headings
            before_blank = i == 0 or not lines[i - 1].strip()
            after_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
            if before_blank and after_blank:
                marks.append((i, line.strip()))

    if len(marks) < 2:
        return [Section(number=1, title="Full text", text=text)]

    sections: list[Section] = []
    # anything before the first heading is front matter; drop it
    for n, (line_no, title) in enumerate(marks, start=1):
        end = marks[n][0] if n < len(marks) else len(lines)
        body = "\n".join(lines[line_no + 1:end]).strip()
        if body:
            sections.append(Section(number=len(sections) + 1, title=title, text=body))

    return sections or [Section(number=1, title="Full text", text=text)]
