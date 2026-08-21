"""Answer validation.

Two layers, deterministic first. The citation check is the one that matters:
it is a set-membership test, not a judgement call, and it is what actually
constrains the model to the evidence it was given.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents import llm
from app.agents.prompts import VALIDATOR_SYSTEM
from app.config import VALIDATOR_USE_MODEL

# Accept any bracketed group, then resolve the ids inside it. Models reliably
# emit the short form "[c00042]" and group several ids as "[c1, c2]", and an
# earlier strict pattern rejected perfectly good answers on formatting alone --
# the retry loop then refused a question it had just answered correctly.
#
# Leniency here is about FORMAT only. Membership stays strict: every resolved
# id must be in the supplied evidence set, which is the property that matters.
_BRACKET = re.compile(r"[\[［【〔⟦〚]"
                      r"([^\]］】〕⟧〛\n]{1,200})"
                      r"[\]］】〕⟧〛]")
_CHUNK_REF = re.compile(r"(?:([A-Za-z0-9_.\-]+)::)?(c\d{4,6})")

# Models intermittently reach for a full-width or CJK bracket instead of the
# ASCII one the prompt asks for -- gpt-oss-120b emits 【c00065】 often enough
# to matter. The citation is CORRECT; only the glyph is not. Under an
# ASCII-only pattern that scored as "no citation at all", which failed
# validation, burned both retries at ~2,700 tokens each, and then refused a
# question the model had already answered from the right passage: three
# calls and a wrong outcome, for a bracket.
#
# Normalising to ASCII before anything parses it fixes the reader's copy
# too, so citations render one way whichever glyph came back.
_BRACKET_VARIANTS = str.maketrans({
    "［": "[", "【": "[", "〔": "[", "⟦": "[", "〚": "[",
    "］": "]", "】": "]", "〕": "]", "⟧": "]", "〛": "]",
})


def normalise_citations(draft: str) -> str:
    """Rewrite full-width/CJK citation brackets to ASCII."""
    return draft.translate(_BRACKET_VARIANTS)


def _resolve(draft: str, allowed: set[str]) -> tuple[list[str], list[str]]:
    """Returns (cited, foreign) chunk ids found in the draft."""
    by_suffix: dict[str, str] = {}
    for full in allowed:
        by_suffix.setdefault(full.rsplit("::", 1)[-1], full)

    cited: list[str] = []
    foreign: list[str] = []
    for group in _BRACKET.findall(draft):
        for book, short in _CHUNK_REF.findall(group):
            full = f"{book}::{short}" if book else ""
            if full not in allowed:
                # Fall back to matching on the chunk suffix. Models copy the
                # book prefix out of the prompt's example, so a correct
                # citation can arrive as "bk::c00000" instead of the real id.
                # Only one book's passages are ever in a prompt, so the suffix
                # is unambiguous -- and this still resolves ONLY into the
                # supplied set, so a fabricated id stays foreign.
                full = by_suffix.get(short, full or short)
            (cited if full in allowed else foreign).append(full)
    return cited, sorted(set(foreign))
# a sentence carrying a factual claim, roughly: has a verb and some length
_TRIVIAL = re.compile(r"^(insufficient|i don't|i cannot|i can't)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    grounded: bool
    reason: str
    cited: list[str]
    foreign: list[str]


def check(draft: str, allowed_ids: list[str], *, use_model: bool = True,
          evidence_text: str = "", evidence=None) -> ValidationResult:
    # An empty draft used to validate as GROUNDED -- no claims, so nothing
    # unsupported -- and the graph then reported outcome "answered" with a
    # blank answer. Vacuous truth is not grounding.
    if not draft.strip():
        return ValidationResult(False, "the model returned an empty answer", [], [])

    allowed = set(allowed_ids)
    cited, foreign = _resolve(draft, allowed)

    if foreign:
        return ValidationResult(
            False, f"cites {len(foreign)} passage(s) not in the evidence set",
            cited, foreign,
        )

    body = _BRACKET.sub("", draft).strip()
    if not _TRIVIAL.match(body):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if len(s.split()) > 3]
        if sentences and not cited:
            return ValidationResult(
                False, "makes claims with no citation at all", cited, foreign)

    if use_model and cited and VALIDATOR_USE_MODEL:
        # Show the validator the FULL text of the passages actually cited.
        #
        # This previously passed every retrieved chunk truncated to 400
        # characters -- about a quarter of a 320-word chunk. The validator
        # then judged, correctly from its own truncated view, that the claims
        # were unsupported, and vetoed answers that were in fact right.
        # Checking cited claims against cited passages is both sounder and
        # cheaper than checking them against a fraction of everything.
        by_id = {e.chunk.id: e.chunk.text for e in (evidence or [])}
        evidence_text = "\n\n".join(
            f"[{cid}] {by_id[cid]}" for cid in dict.fromkeys(cited) if cid in by_id
        ) or evidence_text
        try:
            verdict = llm.chat(
                VALIDATOR_SYSTEM,
                f"Evidence:\n{evidence_text}\n\nDraft answer:\n{draft}",
                role="fast", temperature=0.0,
            ).text.strip().upper()
            if verdict.startswith("UNGROUNDED"):
                return ValidationResult(
                    False, "model judged the draft unsupported", cited, foreign)
        except llm.LLMError:
            pass  # deterministic layer already passed; do not fail closed on transport

    return ValidationResult(True, "grounded", cited, foreign)
