"""Question intent classification. Cheap model; a wrong guess costs recall,
not safety."""
from __future__ import annotations

from app.agents import llm
from app.agents.prompts import ANALYZER_SYSTEM
from app.models import Intent

_VALID = {"character", "event", "motivation", "relationship",
          "summary", "clue", "prediction"}

_HINTS = (
    ("prediction", ("will ", "going to happen", "predict", "next")),
    ("motivation", ("why ", "motive", "reason")),
    ("relationship", ("related", "relationship", "know each other")),
    ("summary", ("summar", "recap", "so far", "catch me up")),
    ("clue", ("clue", "significan", "mean when", "symbol")),
    ("character", ("who is", "who's", "who was")),
)


def classify(question: str) -> Intent:
    q = question.lower()
    for intent, needles in _HINTS:
        if any(n in q for n in needles):
            return intent  # type: ignore[return-value]
    try:
        word = llm.chat(ANALYZER_SYSTEM, question, role="fast", temperature=0.0).text.strip().lower()
        if word in _VALID:
            return word  # type: ignore[return-value]
    except llm.LLMError:
        pass
    return "event"
