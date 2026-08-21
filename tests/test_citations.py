"""Citation parsing: the format is lenient, the membership is not.

The distinction is the whole point. A model that cites the right passage in
the wrong glyph has answered correctly and must be accepted; a model that
cites an id it was never given must be rejected however well-formed the
brackets are. These tests pin both halves, because the failure mode is
silent -- a rejected-on-format answer comes back as "not enough evidence",
which looks exactly like correct spoiler-safe behaviour.
"""
from __future__ import annotations

import pytest

from app.agents.validator import check, normalise_citations

ALLOWED = ["bk::c00065", "bk::c00066"]


@pytest.mark.parametrize("opener,closer", [
    ("[", "]"),          # ASCII, what the prompt asks for
    ("［", "］"),  # full-width
    ("【", "】"),  # CJK black lenticular -- gpt-oss-120b reaches for these
    ("〔", "〕"),  # tortoise shell
    ("⟦", "⟧"),  # mathematical white square
])
def test_citation_accepted_in_any_bracket_glyph(opener, closer):
    """A correct citation in an unexpected bracket is still a citation.

    Before this, an answer citing 【c00065】 scored as "makes claims with no
    citation at all", failed validation, spent both retries re-answering,
    and refused a question it had already got right.
    """
    draft = f"Henry Clerval is Victor's friend {opener}c00065{closer}."
    result = check(draft, ALLOWED, use_model=False)
    assert result.grounded, result.reason
    assert result.cited == ["bk::c00065"]


def test_normalise_rewrites_variants_to_ascii():
    draft = "One 【c00065】. Two ［c00066］."
    assert normalise_citations(draft) == "One [c00065]. Two [c00066]."


def test_normalise_leaves_ordinary_prose_alone():
    draft = "Victor studies chemistry [c00065]."
    assert normalise_citations(draft) == draft


def test_fabricated_id_stays_foreign_in_any_glyph():
    """Leniency is about FORMAT only -- membership is still a set test."""
    for opener, closer in (("[", "]"), ("【", "】")):
        result = check(f"Invented {opener}c99999{closer}.", ALLOWED, use_model=False)
        assert not result.grounded
        assert result.foreign == ["c99999"]


def test_uncited_claim_still_fails():
    """The variants must not accidentally make everything look cited."""
    result = check("Henry Clerval is Victor's friend.", ALLOWED, use_model=False)
    assert not result.grounded
    assert "no citation" in result.reason
