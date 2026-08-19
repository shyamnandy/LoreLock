"""List the models your configured provider is actually serving.

Model IDs churn -- Groq deprecated kimi-k2, qwen3-32b and llama-4-scout
during 2026 -- so hardcoding one eventually 404s. Run this, pick real IDs,
and set LLM_MODEL_ANSWER / LLM_MODEL_FAST from the output.

    python scripts/models.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import model_for  # noqa: E402
from app.agents import llm        # noqa: E402

ENDPOINTS = {
    "groq": ("https://api.groq.com/openai/v1/models", "GROQ_API_KEY", "bearer"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models",
               "GEMINI_API_KEY", "goog"),
    "ollama": (f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api/tags",
               None, None),
}


def fetch(provider: str) -> list[str]:
    if provider not in ENDPOINTS:
        raise SystemExit(f"no model listing for provider {provider!r}")
    url, env_key, scheme = ENDPOINTS[provider]

    # same Cloudflare guard as app/agents/llm.py
    headers = {"User-Agent": "spoiler-safe-reading-companion/0.1",
               "Accept": "application/json"}
    if env_key:
        key = os.environ.get(env_key)
        if not key:
            raise SystemExit(f"{env_key} is not set (put it in .env)")
        headers |= ({"Authorization": f"Bearer {key}"} if scheme == "bearer"
                    else {"x-goog-api-key": key})

    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=30
        ) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{exc.code}: {exc.read().decode()[:300]}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc.reason}") from exc

    if provider == "ollama":
        return sorted(m["name"] for m in data.get("models", []))
    if provider == "gemini":
        return sorted(m["name"].removeprefix("models/") for m in data.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))
    return sorted(m["id"] for m in data.get("data", []))


if __name__ == "__main__":
    provider = llm.provider_name()
    if provider == "stub":
        raise SystemExit("LLM_PROVIDER is 'stub' -- set groq, gemini or ollama first")

    live = fetch(provider)
    answer, fast = model_for("answer", provider), model_for("fast", provider)

    print(f"\n  {provider} is serving {len(live)} models\n")
    for m in live:
        tags = []
        if m == answer:
            tags.append("<- answer node")
        if m == fast:
            tags.append("<- fast node")
        print(f"    {m:<44}{' '.join(tags)}")

    missing = [r for r, m in (("answer", answer), ("fast", fast)) if m not in live]
    if missing:
        print(f"\n  WARNING: configured model(s) not in the live list: "
              f"{', '.join(missing)}")
        print("  Set LLM_MODEL_ANSWER / LLM_MODEL_FAST to IDs from above.")
    else:
        print("\n  both configured models are live")
