"""Render an interview Q&A markdown file to a printable PDF.

Deliberately a renderer over the markdown file rather than a second copy of
the content: the .md stays the single source of truth, and regenerating the
PDF is one command after an edit.

Handles the small markdown subset the Q&A files actually use -- ATX headings,
paragraphs, horizontal rules, ``` fenced blocks, and inline **bold** /
*italic* / `code`. The `->` aside lines (the "only if they push" detail) get
their own tinted style, since that distinction is the whole point of the
document's layout.

    python scripts/generate_qa_pdf.py                       # the 50-question set
    python scripts/generate_qa_pdf.py INTERVIEW_QA_107.md out.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "INTERVIEW_QA_50.md"
OUT = ROOT / "output" / "pdf" / "BOOKMAte_50_interview_questions.pdf"

INK = colors.HexColor("#123047")
MUTED = colors.HexColor("#52616b")
RULE = colors.HexColor("#c9d4dc")
CODE_BG = colors.HexColor("#eef2f5")
ASIDE_BG = colors.HexColor("#f4f7f9")

_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def inline(text: str) -> str:
    """Markdown inline -> reportlab mini-HTML, escaping first so book text
    containing < or & cannot break the tag stream."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # the built-in Helvetica has no arrow glyph; it would render as a box
    if text.startswith("→"):
        text = "<b>If pushed:</b> " + text[1:].lstrip()
    text = _CODE.sub(
        lambda m: f'<font face="Courier" size="9" backColor="#eef2f5">{m.group(1)}</font>',
        text,
    )
    text = _BOLD.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _ITALIC.sub(lambda m: f"<i>{m.group(1)}</i>", text)
    return text


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.6,
        leading=13.4, textColor=colors.HexColor("#1d2b36"), spaceAfter=7,
    )
    return {
        "title": ParagraphStyle(
            "TitleCenter", parent=base["Title"], alignment=TA_CENTER,
            fontName="Helvetica-Bold", fontSize=21, leading=26,
            textColor=INK, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2x", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=14, leading=17, textColor=INK,
            spaceBefore=14, spaceAfter=8,
        ),
        "q": ParagraphStyle(
            "Question", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=10.6, leading=14, textColor=colors.HexColor("#0f4c75"),
            spaceBefore=11, spaceAfter=4,
        ),
        "body": body,
        "aside": ParagraphStyle(
            "Aside", parent=body, fontSize=9, leading=12.6, textColor=MUTED,
            leftIndent=10, rightIndent=6, spaceBefore=2, spaceAfter=9,
            borderPadding=(5, 5, 5, 5), backColor=ASIDE_BG,
            borderColor=RULE, borderWidth=0.5,
        ),
        "code": ParagraphStyle(
            "Code", fontName="Courier", fontSize=7.2, leading=9,
            textColor=colors.HexColor("#1d2b36"), backColor=CODE_BG,
            borderPadding=(6, 6, 6, 6), spaceBefore=4, spaceAfter=10,
        ),
    }


def blocks(md: str) -> list[tuple[str, str]]:
    """Split the file into (kind, text) blocks, unwrapping soft line breaks."""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    kind = "body"

    def flush() -> None:
        nonlocal buf, kind
        if buf:
            out.append((kind, " ".join(buf).strip()))
        buf, kind = [], "body"

    code: list[str] | None = None
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if code is None:
                flush()
                code = []
            else:
                out.append(("code", "\n".join(code)))
                code = None
            continue
        if code is not None:
            code.append(line)   # verbatim: a diagram's alignment is its meaning
        elif not stripped:
            flush()
        elif stripped == "---":
            flush()
            out.append(("rule", ""))
        elif stripped.startswith("### "):
            flush()
            out.append(("q", stripped[4:]))
        elif stripped.startswith("## "):
            flush()
            out.append(("h2", stripped[3:]))
        elif stripped.startswith("# "):
            flush()
            out.append(("title", stripped[2:]))
        elif stripped.startswith("→"):
            flush()
            kind = "aside"
            buf.append(stripped)
        else:
            buf.append(stripped)
    flush()
    return out


def story(md: str):
    st = styles()
    flow: list = []
    pending_q: Paragraph | None = None

    for kind, text in blocks(md):
        if kind == "rule":
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
            flow.append(Spacer(1, 2))
            continue
        if kind == "code":
            para = Preformatted(text, st["code"])
            if pending_q is not None:
                flow.append(KeepTogether([pending_q, para]))
                pending_q = None
            else:
                flow.append(para)
            continue
        para = Paragraph(inline(text), st[kind if kind in st else "body"])
        if kind == "q":
            # hold the heading back so a question never lands alone at the
            # foot of a page, separated from its own answer
            pending_q = para
            continue
        if pending_q is not None:
            flow.append(KeepTogether([pending_q, para]))
            pending_q = None
        else:
            flow.append(para)
    if pending_q is not None:
        flow.append(pending_q)
    return flow


def footer(label: str):
    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#687782"))
        canvas.drawString(0.65 * inch, 0.42 * inch, label)
        canvas.drawRightString(A4[0] - 0.65 * inch, 0.42 * inch, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    md = src.read_text(encoding="utf-8")
    # the document's own H1 names the PDF, so a new Q&A file needs no edit here
    title = next((ln[2:].strip() for ln in md.splitlines() if ln.startswith("# ")),
                 "BookMate - Interview Questions")
    label = title.split(",")[0].replace("—", "-")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        rightMargin=0.68 * inch, leftMargin=0.68 * inch,
        topMargin=0.62 * inch, bottomMargin=0.65 * inch,
        title=title, author="Shyam",
    )
    doc.build(story(md), onFirstPage=footer(label), onLaterPages=footer(label))
    print(out)


if __name__ == "__main__":
    main()
