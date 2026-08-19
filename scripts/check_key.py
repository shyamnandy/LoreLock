"""Confirm the provider key is loaded, without printing it.

    python scripts/check_key.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agents import llm            # noqa: E402  (imports app.config -> loads .env)
from app.config import ROOT, model_for  # noqa: E402

env = ROOT / ".env"
print(f"  .env file        {'found' if env.exists() else 'MISSING at ' + str(env)}")

provider = llm.provider_name()
print(f"  LLM_PROVIDER     {provider}")
if provider == "stub":
    sys.exit("\n  Still on the keyless stub. Answers will be quoted passages,\n"
             "  not synthesis. Create .env with LLM_PROVIDER and your key.")

print(f"  answer model     {model_for('answer', provider)}")
print(f"  fast model       {model_for('fast', provider)}")

try:
    reply = llm.chat("Reply with exactly: OK", "ping", role="fast")
except llm.LLMError as exc:
    sys.exit(f"\n  call FAILED -> {exc}")
print(f"  live call        {reply.text[:40]!r} via {reply.model}")
print("\n  ready")
