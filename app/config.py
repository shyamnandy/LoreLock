"""Central configuration. Everything tunable lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment.

    Deliberately dependency-free and non-overriding: a variable already set
    in the real environment always wins, so CI and shell exports are not
    silently shadowed by a stale file on disk.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")
# DATA_DIR lets a container point the writable state at a mounted disk.
# Without it a redeploy would silently start from an empty index.
DATA = Path(os.environ.get("DATA_DIR") or (ROOT / "data"))
RAW = DATA / "raw"
DB_PATH = DATA / "db" / "companion.sqlite3"
VEC_DIR = DATA / "db" / "vectors"

for _d in (RAW, DB_PATH.parent, VEC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- position model -------------------------------------------------------
# word offset is ground truth; pages are a synthetic view over it.
WORDS_PER_PAGE = 250

# --- chunking -------------------------------------------------------------
# 320 rather than 200: a narrative answer needs surrounding context, and the
# screenshot evidence showed 200-word passages cutting off mid-exchange.
CHUNK_WORDS = 320
CHUNK_OVERLAP = 80

# --- embeddings -----------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# --- retrieval ------------------------------------------------------------
# 5 not 8: each chunk is now ~320 words, and neighbour expansion triples the
# count. Eight seeds alone would blow the free-tier token budget.
TOP_K = 5
RRF_K = 60

# raw cosine, calibrated against bge-small: relevant passages land ~0.60-0.70,
# clearly off-topic queries ~0.40-0.46. NOT an RRF score -- see graph.assess_evidence.
# A CHEAP GATE ONLY -- not a relevance judgement.
#
# Measured on both corpora, neither absolute cosine nor a z-score against the
# eligible corpus separates "this book can answer the question" from "it
# cannot": bge-small scores 0.455 for "capital of Peru" against Sherlock
# Holmes, and the novella's off-topic control out-scores its real questions.
# So this floor only screens out the degenerate cases, and the real
# sufficiency judgement is made by the answer agent, which can actually read
# the passages. See app/graph.py -> route_after_validate.
SUFFICIENCY_FLOOR = 0.30
# the oracle calls "answered later" when future material beats this cosine
ORACLE_THRESHOLD = 0.62

# --- answering ------------------------------------------------------------
MAX_VALIDATION_RETRIES = 2

# --- model routing --------------------------------------------------------
#
# Route per NODE, not globally. The answer node is the only one whose output
# a reader sees, and the only one that must decline rather than guess -- give
# it the capable model. Intent classification and the entailment check are
# high-volume, low-difficulty calls where a small fast model is strictly
# better: comparable accuracy, a fraction of the latency and quota.
#
# Model IDs churn (Groq deprecated kimi-k2, qwen3-32b and llama-4-scout
# during 2026), so these are overridable and `scripts/models.py` lists what
# the configured provider is actually serving today.
_DEFAULT_MODELS = {
    # verified live 2026-08-18 via scripts/models.py. The llama-3.x chat
    # models were withdrawn from Groq's self-serve catalogue, which is the
    # whole reason that script exists -- check before trusting these.
    "groq":   {"answer": "openai/gpt-oss-120b",     "fast": "openai/gpt-oss-20b"},
    "gemini": {"answer": "gemini-2.0-flash",        "fast": "gemini-2.0-flash-lite"},
    "ollama": {"answer": "llama3.1",                "fast": "llama3.2:3b"},
    "stub":   {"answer": "",                        "fast": ""},
}


def model_for(role: str, provider: str) -> str:
    """Resolve the model for a node role.

    Precedence: LLM_MODEL_<ROLE> -> LLM_MODEL -> provider default.
    """
    override = (os.environ.get(f"LLM_MODEL_{role.upper()}")
                or os.environ.get("LLM_MODEL"))
    if override:
        return override
    return _DEFAULT_MODELS.get(provider, {}).get(role, "")

# --- evidence budget ------------------------------------------------------
#
# Free tiers cap tokens per minute (Groq: 8,000 TPM), and neighbour expansion
# multiplies passage count by up to 3. Without a budget the answer call fails
# with HTTP 413 -- which used to surface as a plain "not enough evidence"
# refusal, making a broken provider look like correct behaviour.
#
# ~4 chars per token, leaving room for the system prompt and the reply.
# ~2,200 tokens per call. Groq free tier allows 8,000 TPM and 200,000 TPD per
# model, so a 4,000-token prompt permits barely one call a minute and burns
# the day in ~45 turns. Smaller passages, more of them, is the better trade.
MAX_EVIDENCE_CHARS = 9_000

# --- validation -----------------------------------------------------------
#
# Two layers guard grounding. The DETERMINISTIC one -- every cited id must be
# in the supplied evidence -- is load-bearing and always runs. The model-based
# entailment check is a second opinion, and it costs a second API call per
# turn, which on a rate-limited free tier roughly halves throughput.
#
# It also has veto power over a correct answer, so it earns its place only
# when it is reliable. Set VALIDATOR_USE_MODEL=0 to run deterministic-only.
VALIDATOR_USE_MODEL = os.environ.get("VALIDATOR_USE_MODEL", "1") not in {"0", "false", "no"}
