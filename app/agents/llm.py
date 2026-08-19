"""Provider-agnostic chat interface.

Deliberately thin. The project's claims are about retrieval and access
control, so every LLM-dependent part must be swappable and the firewall
must be testable with no provider at all -- hence the `stub` backend.

Select with LLM_PROVIDER: gemini | groq | ollama | stub
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# imported for its side effect: app.config loads .env into os.environ before
# any backend below reads a key. Without this, importing llm directly would
# see an unpopulated environment.
import app.config  # noqa: F401


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Reply:
    text: str
    provider: str
    model: str = ""
    role: str = "answer"


# urllib's default User-Agent is "Python-urllib/3.x", which Cloudflare blocks
# outright in front of some provider APIs -- Groq returns 403 "error code:
# 1010" with a perfectly valid key. Always identify properly.
USER_AGENT = "spoiler-safe-reading-companion/0.1 (+https://github.com/)"

MAX_RATE_LIMIT_RETRIES = 4
_RETRY_HINT = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)


def _retry_after(exc, body: str, cap: float = 30.0) -> float:
    """How long to wait after a 429.

    Prefer the provider's own answer -- the Retry-After header, else the
    "please try again in 13.695s" hint in the body -- and only fall back to
    a guess. Waiting the stated time succeeds first try; guessing does not.
    """
    header = exc.headers.get("retry-after") if getattr(exc, "headers", None) else None
    if header:
        try:
            return min(float(header) + 0.5, cap)
        except ValueError:
            pass
    if m := _RETRY_HINT.search(body):
        return min(float(m.group(1)) + 0.5, cap)
    return 5.0


def _post(url: str, payload: dict, headers: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": USER_AGENT,
                 "Accept": "application/json",
                 **headers},
    )
    # Free tiers are rate-limited per MINUTE (Groq: 8,000 TPM), so a burst of
    # otherwise-valid calls gets 429s. The provider tells us how long to wait;
    # honouring that is the difference between a working eval run and a page
    # of errors.
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:400]
            if exc.code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                time.sleep(_retry_after(exc, body))
                continue
            raise LLMError(f"{exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"cannot reach {url}: {exc.reason}") from exc
    raise LLMError("exhausted rate-limit retries")  # pragma: no cover


# -- backends --------------------------------------------------------------

def _gemini(system: str, user: str, model: str, temperature: float) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    model = model or os.environ.get("LLM_MODEL", "gemini-2.0-flash")
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 1024},
        },
        {"x-goog-api-key": key},
    )
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected Gemini response: {str(data)[:300]}") from exc


def _groq(system: str, user: str, model: str, temperature: float) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set")
    model = model or os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 1024,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    # gpt-oss models reason before answering and return `reasoning` alongside
    # `content`. Left on the default effort with a long evidence prompt they
    # can spend the whole budget thinking and return EMPTY content -- which
    # reached the reader as a blank answer. Grounded extraction from supplied
    # passages needs little deliberation, so ask for little.
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"

    data = _post("https://api.groq.com/openai/v1/chat/completions", payload,
                 {"Authorization": f"Bearer {key}"})

    message = data["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    if not content:
        # last resort: better a reasoning trace than a blank reply, and the
        # citation checker will reject it if it is not properly grounded
        content = (message.get("reasoning") or "").strip()
    if not content:
        raise LLMError(
            f"{model} returned an empty message "
            f"(finish_reason={data['choices'][0].get('finish_reason')})"
        )
    return content


def _ollama(system: str, user: str, model: str, temperature: float) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.environ.get("LLM_MODEL", "llama3.1")
    data = _post(
        f"{host}/api/chat",
        {"model": model, "stream": False, "options": {"temperature": temperature},
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}]},
        {},
    )
    return data["message"]["content"]


_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have he her him "
    "his how i in is it its of on or she that the their them they this to was "
    "were what when where which who whom why will with you your".split()
)


def _stub(system: str, user: str, model: str, temperature: float) -> str:
    """Keyless test double.

    Quotes the best-matching supplied passage and cites it, so the whole graph
    -- retrieval, citation checking, the validator cycle -- runs offline with
    no key. It also *declines* when the passages share too little vocabulary
    with the question, because a double that can never say INSUFFICIENT would
    leave the refusal branches unexercised.

    It is not a language model and does not pretend to be one; it exists so
    the firewall's guarantees can be tested without a provider.
    """
    marker = "[chunk "
    if marker not in user:
        return "INSUFFICIENT"

    # the question trails the final evidence block, so split it off FIRST --
    # otherwise the last passage is scored against the question's own words
    # and always wins.
    evidence_part, _, question = user.rpartition("Reader's question:")
    question = question.strip().lower()
    terms = {w for w in re.findall(r"[a-z']+", question)
             if len(w) > 2 and w not in _STOPWORDS}

    best, best_hits, best_cid = "", 0, ""
    for block in evidence_part.split(marker)[1:]:
        cid, _, body = block.partition("]")
        body = body.strip()
        hits = sum(1 for t in terms if t in body.lower())
        if hits > best_hits:
            best, best_hits, best_cid = body, hits, cid.strip()

    if best_hits < 2:
        return "INSUFFICIENT"

    sentences = re.split(r"(?<=[.!?])\s+", best)
    return f"{' '.join(sentences[:2])[:280].strip()} [{best_cid}]"

_BACKENDS = {"gemini": _gemini, "groq": _groq, "ollama": _ollama, "stub": _stub}


def provider_name() -> str:
    return os.environ.get("LLM_PROVIDER", "stub").lower()


def available() -> bool:
    name = provider_name()
    if name == "stub":
        return True
    return bool(os.environ.get({"gemini": "GEMINI_API_KEY",
                                "groq": "GROQ_API_KEY"}.get(name, "OLLAMA_HOST"))
                or name == "ollama")


def chat(system: str, user: str, *, role: str = "answer", model: str = "",
         temperature: float = 0.0) -> Reply:
    """Call the configured provider.

    `role` selects which model this node gets -- "answer" for the reader-facing
    synthesis, "fast" for classification and validation. An explicit `model`
    always wins.
    """
    name = provider_name()
    backend = _BACKENDS.get(name)
    if backend is None:
        raise LLMError(f"unknown LLM_PROVIDER {name!r}; expected one of {sorted(_BACKENDS)}")
    chosen = model or app.config.model_for(role, name)
    return Reply(text=backend(system, user, chosen, temperature).strip(),
                 provider=name, model=chosen or name, role=role)
