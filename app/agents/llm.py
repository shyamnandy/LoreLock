"""Provider-agnostic chat interface.

Deliberately thin. The project's claims are about retrieval and access
control, so every LLM-dependent part must be swappable and the firewall
must be testable with no provider at all -- hence the `stub` backend.

Select with LLM_PROVIDER: gemini | groq | ollama | stub
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
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
# "please try again in 13.695s", and the daily-quota form, "in 9m33.696s".
# The minutes group is not decoration: without it the longer form parses as
# 33 seconds, and a limit that clears tomorrow looks like one that clears
# before the reader loses patience.
_RETRY_HINT = re.compile(r"try again in (?:(\d+)m)?([\d.]+)\s*s", re.IGNORECASE)


def _retry_after(exc, body: str, cap: float = 30.0) -> float | None:
    """How long to wait after a 429, or None to stop retrying.

    Prefer the provider's own answer -- the Retry-After header, else the
    "please try again in 13.695s" hint in the body -- and only fall back to
    a guess. Waiting the stated time succeeds first try; guessing does not.

    A stated wait longer than `cap` is the tell that this is not a
    per-minute burst limit but a daily quota (Groq: 200,000 TPD per model),
    and no number of retries inside one request will clear it. Sleeping the
    cap four times over turns a failure into a two-minute failure, which is
    strictly worse -- the reader waits and still gets nothing. Give up at
    once and let the caller surface it.
    """
    wait = None
    header = exc.headers.get("retry-after") if getattr(exc, "headers", None) else None
    if header:
        try:
            wait = float(header)
        except ValueError:
            pass
    if wait is None:
        if m := _RETRY_HINT.search(body):
            wait = float(m.group(1) or 0) * 60 + float(m.group(2))
        else:
            return 5.0
    return wait + 0.5 if wait + 0.5 <= cap else None


# --- per-model token budget ----------------------------------------------
#
# Groq meters tokens per MINUTE and per MODEL, and reports the state of that
# meter on every response. Firing concurrent requests blind and dealing with
# the 429s afterwards wastes the window twice over: the losing request has
# already paid its full latency before being told no, and the retry then
# sleeps through the part of the minute it could have been using.
#
# Tracking the meter locally lets a caller queue BEFORE it spends, which is
# what turns "two readers asked at once" from a 429 into a short wait. The
# provider's headers stay authoritative -- this is only a forecast for the
# gap between responses, so it starts pessimistic and is corrected by every
# reply that comes back.
#
# Per model, not global, because the buckets are independent: routing
# `answer` and `fast` at different models is worth 2x the throughput of
# putting both on one, and this is the bookkeeping that makes that real.
TPM_LIMIT = int(os.environ.get("LLM_TPM_LIMIT", "8000"))

_REFILL_RATE = TPM_LIMIT / 60.0                # tokens per second

_budget_lock = threading.Lock()
_budget: dict[str, tuple[float, float]] = {}   # model -> (tokens_left, measured_at)

def _estimate(payload: dict) -> int:
    """Tokens this call will be billed, near enough to reserve against.

    ~4 characters per token for English prose, plus the whole completion
    allowance -- reasoning tokens are billed as completion, so a model that
    thinks its way through the budget costs exactly max_tokens.
    """
    chars = sum(len(m.get("content") or "") for m in payload.get("messages", []))
    return chars // 4 + int(payload.get("max_tokens") or 1024)


def _reserve(model: str, estimate: int, cap: float = 30.0) -> None:
    """Block until `estimate` tokens fit this model's allowance.

    The bucket refills CONTINUOUSLY -- 8,000 TPM is 133 tokens a second, not
    8,000 tokens handed over on the minute. Waiting for a window boundary
    would idle through a refill that is already happening and turn three
    concurrent questions into three minutes; waiting only for the deficit
    turns them into the ~20s they actually cost.

    This bounds throughput at TPM/tokens_per_question and nothing here can
    raise that ceiling -- it exists so that reaching the ceiling degrades
    into a queue instead of a burst of 429s.
    """
    while True:
        with _budget_lock:
            now = time.monotonic()
            left, measured_at = _budget.get(model, (float(TPM_LIMIT), now))
            left = min(float(TPM_LIMIT), left + (now - measured_at) * _REFILL_RATE)
            if left >= estimate:
                _budget[model] = (left - estimate, now)
                return
            wait = (estimate - left) / _REFILL_RATE
            _budget[model] = (left, now)
        # Outside the lock: a thread sleeping on it would serialise exactly
        # what this exists to overlap.
        time.sleep(min(max(wait, 0.05), cap))


def _observe(model: str, resp_headers) -> None:
    """Replace the forecast with what the provider actually reports."""
    remaining = resp_headers.get("x-ratelimit-remaining-tokens")
    if remaining is None:
        return
    try:
        left = float(remaining)
    except ValueError:
        return
    # The header is ground truth and supersedes whatever the forecast held,
    # in both directions: it releases the slack left over from a call that
    # reserved max_tokens and used a fifth of it, and it tightens up when
    # another process shares the same account.
    with _budget_lock:
        _budget[model] = (left, time.monotonic())


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
    model = payload.get("model", "")
    estimate = _estimate(payload)
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        _reserve(model, estimate)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                _observe(model, resp.headers)
                return data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:400]
            if exc.code == 429:
                _observe(model, exc.headers)
            if exc.code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                delay = _retry_after(exc, body)
                if delay is not None:
                    time.sleep(delay)
                    continue
            raise LLMError(f"{exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"cannot reach {url}: {exc.reason}") from exc
    raise LLMError("exhausted rate-limit retries")  # pragma: no cover


# -- backends --------------------------------------------------------------

def _gemini(system: str, user: str, model: str, temperature: float,
            max_tokens: int) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    model = model or os.environ.get("LLM_MODEL", "gemini-2.0-flash")
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        },
        {"x-goog-api-key": key},
    )
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected Gemini response: {str(data)[:300]}") from exc


def _groq(system: str, user: str, model: str, temperature: float,
          max_tokens: int) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set")
    model = model or os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
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


def _ollama(system: str, user: str, model: str, temperature: float,
            max_tokens: int) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.environ.get("LLM_MODEL", "llama3.1")
    data = _post(
        f"{host}/api/chat",
        {"model": model, "stream": False,
         "options": {"temperature": temperature, "num_predict": max_tokens},
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


def _stub(system: str, user: str, model: str, temperature: float,
          max_tokens: int = 0) -> str:
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


# -- keys ------------------------------------------------------------------
#
# Which environment variable each provider reads its key from. `ollama` and
# `stub` are absent rather than mapped to "": they need no key at all, and a
# blank entry would read as "a key that happens to be empty".
KEY_VAR = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}

# The key this process STARTED with -- an export, or .env via app.config at
# import time. Everything below writes the live key straight into os.environ,
# because that is the only thing the backends read; but an override with no
# snapshot behind it is a one-way door. Keep the original so clearing the
# override returns the deployment to its OWN key rather than to no key.
_BASELINE_KEYS = {var: os.environ.get(var) for var in KEY_VAR.values()}

# os.environ is process-global and a request may arrive mid-swap. The lock
# does not make the override per-user -- it cannot; there is one environment
# -- it just keeps a read from landing between the delete and the set.
_key_lock = threading.Lock()


def key_var(provider: str = "") -> str:
    """The env var holding the key for `provider`, or "" if it needs none."""
    return KEY_VAR.get(provider or provider_name(), "")


def key_hint(key: str) -> str:
    """A key rendered safe to return over the wire.

    The last four characters only. Enough to tell two keys apart -- which is
    the entire question a reader has ("is it using the one I pasted?") --
    and not enough to use.
    """
    return f"...{key[-4:]}" if len(key) > 4 else "..."


def key_state(provider: str = "") -> dict:
    """Where the live key came from, without disclosing it.

    source is one of:
      none        -- this provider needs no key (stub, ollama)
      missing     -- it needs one and there is none
      environment -- the key the deployment was configured with
      override    -- a key pasted at runtime, replacing that one
    """
    provider = provider or provider_name()
    var = KEY_VAR.get(provider)
    if not var:
        return {"provider": provider, "source": "none", "hint": ""}
    live = os.environ.get(var) or ""
    if not live:
        return {"provider": provider, "source": "missing", "hint": ""}
    source = "environment" if live == _BASELINE_KEYS.get(var) else "override"
    return {"provider": provider, "source": source, "hint": key_hint(live)}


# -- remembering the pasted key -------------------------------------------
#
# os.environ alone lasts as long as the process, and a free instance restarts
# every time it sleeps -- so a key pasted once would have to be pasted again
# after every nap. It is written to app.config.KEY_STORE instead and restored
# at startup.
#
# What is stored is the key plus a FINGERPRINT of the configured key it was
# overriding -- not that key itself, which has no business being written to
# disk a second time. The fingerprint answers one question at startup: has the
# operator changed the deployment's own key since this override was saved? If
# so the override is stale by definition -- someone rotated the real key and
# redeployed -- and it is discarded rather than allowed to shadow the new one.


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _read_store() -> dict:
    try:
        data = json.loads(app.config.KEY_STORE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        # unreadable or corrupt is not fatal: it means no saved key, and the
        # process falls back to the configured one exactly as it would have
        return {}


def _write_store(data: dict) -> None:
    path = app.config.KEY_STORE
    tmp = path.with_suffix(".tmp")
    try:
        if not data:
            path.unlink(missing_ok=True)
            return
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # before the rename, so the file is never briefly world-readable
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass                      # windows has no mode bits to set
        os.replace(tmp, path)         # atomic: no half-written key on disk
    except OSError:
        # persistence is a convenience. A read-only DATA_DIR must not turn
        # a working paste into an error.
        tmp.unlink(missing_ok=True)


def restore_saved_keys() -> list[str]:
    """Reinstall keys saved by an earlier run. Returns the providers restored.

    Called once at startup, after app.config has loaded .env, so
    `_BASELINE_KEYS` already holds whatever this deployment is configured
    with and the comparison below is against the real thing.
    """
    if not app.config.PERSIST_API_KEY:
        return []
    store, restored, changed = _read_store(), [], False
    for provider, var in KEY_VAR.items():
        rec = store.get(var)
        if not isinstance(rec, dict) or not rec.get("key"):
            continue
        if rec.get("over") != _fingerprint(_BASELINE_KEYS.get(var) or ""):
            # the configured key moved underneath this override; drop it
            store.pop(var)
            changed = True
            continue
        os.environ[var] = rec["key"]
        restored.append(provider)
    if changed:
        _write_store(store)
    return restored


def verify_key(key: str, provider: str = "") -> None:
    """Raise LLMError unless `key` authenticates against `provider`.

    A model listing, not a completion: it costs no tokens and does not touch
    the per-minute budget, so checking a key never spends part of the quota
    the reader pasted it to get back.
    """
    provider = provider or provider_name()
    key = key.strip()
    if not key:
        raise LLMError("no key given")
    if provider not in KEY_VAR:
        raise LLMError(f"provider {provider!r} takes no API key")

    url, header = {
        "groq": ("https://api.groq.com/openai/v1/models",
                 {"Authorization": f"Bearer {key}"}),
        "gemini": ("https://generativelanguage.googleapis.com/v1beta/models",
                   {"x-goog-api-key": key}),
    }[provider]
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"} | header)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read(1)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise LLMError(f"{provider} rejected that key") from exc
        # A 429 here means the key is real and the ACCOUNT is rate limited,
        # which is not a reason to refuse to install it.
        if exc.code == 429:
            return
        raise LLMError(f"{provider} returned {exc.code} checking the key") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"cannot reach {provider}: {exc.reason}") from exc


def set_api_key(key: str, provider: str = "", *, persist: bool = True) -> dict:
    """Install `key` as the live key for `provider`. Verified by the caller."""
    provider = provider or provider_name()
    var = KEY_VAR.get(provider)
    if not var:
        raise LLMError(f"provider {provider!r} takes no API key")
    key = key.strip()
    with _key_lock:
        os.environ[var] = key
        if persist and app.config.PERSIST_API_KEY:
            _write_store(_read_store() | {
                var: {"key": key,
                      "over": _fingerprint(_BASELINE_KEYS.get(var) or "")}})
    return key_state(provider)


def clear_api_key(provider: str = "") -> dict:
    """Drop any runtime override and fall back to the configured key.

    Restores the value captured at import -- so a host that was deployed with
    a working key keeps working after someone else's pasted key is cleared --
    and unsets the variable entirely when there was no configured key to
    return to, rather than leaving an empty string that reads as "set".
    """
    provider = provider or provider_name()
    var = KEY_VAR.get(provider)
    if not var:
        raise LLMError(f"provider {provider!r} takes no API key")
    baseline = _BASELINE_KEYS.get(var)
    with _key_lock:
        if baseline:
            os.environ[var] = baseline
        else:
            os.environ.pop(var, None)
        # forget it on disk too, or the next restart would bring it back and
        # "use the server's key instead" would be a lie that lasted one boot
        if (store := _read_store()).pop(var, None) is not None:
            _write_store(store)
    return key_state(provider)


def available() -> bool:
    name = provider_name()
    if name == "stub":
        return True
    if name not in KEY_VAR:
        return True  # ollama and friends authenticate by reachability
    return bool(os.environ.get(KEY_VAR[name]))


def chat(system: str, user: str, *, role: str = "answer", model: str = "",
         temperature: float = 0.0, max_tokens: int = 0) -> Reply:
    """Call the configured provider.

    `role` selects which model this node gets -- "answer" for the reader-facing
    synthesis, "fast" for classification and validation. An explicit `model`
    always wins.

    `role` also sets the completion allowance, since the two travel together:
    the node that may write four sentences is the node that needs room for
    them, and the one replying GROUNDED does not.
    """
    name = provider_name()
    backend = _BACKENDS.get(name)
    if backend is None:
        raise LLMError(f"unknown LLM_PROVIDER {name!r}; expected one of {sorted(_BACKENDS)}")
    chosen = model or app.config.model_for(role, name)
    budget = max_tokens or app.config.MAX_COMPLETION_TOKENS.get(role, 1024)
    return Reply(text=backend(system, user, chosen, temperature, budget).strip(),
                 provider=name, model=chosen or name, role=role)
