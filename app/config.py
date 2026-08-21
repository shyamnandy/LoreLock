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

# Where a key pasted from the UI is remembered, so it survives a restart
# rather than lasting only as long as the process. Inside data/db, which is
# already gitignored and dockerignored, and which scripts/build_seed.py never
# sweeps up -- it copies the sqlite file and *.npy explicitly, so a stray
# credential cannot ride along into the committed seed.
#
# On a host with no mounted disk this is as ephemeral as everything else
# under DATA; the browser re-arms the key there. Set PERSIST_API_KEY=0 to
# keep pasted keys in memory only, which is what a public deployment that
# strangers can reach should do.
KEY_STORE = DB_PATH.parent / "provider_keys.json"
PERSIST_API_KEY = os.environ.get("PERSIST_API_KEY", "1") not in {"0", "false", "no"}

# A read-only, prebuilt library baked into the image. On a host with no
# persistent disk DATA_DIR is empty on every boot, so without this the
# deployed API comes up with no books and every question 404s. Public-domain
# titles only -- see scripts/build_seed.py. Point SEED_DIR at nothing to
# disable, which is what a host with a real disk should do.
SEED_DIR = Path(os.environ.get("SEED_DIR") or (ROOT / "seed"))

# Whether this host may build an index, as opposed to only serving one.
#
# Ingesting is minutes of pegged CPU and the app's memory high-water mark
# (CHALLENGES.md 6). On a fractional-core free instance it is the one
# operation that reliably gets the container throttled or restarted -- and
# on a host with no disk the result is thrown away on the next boot anyway,
# so the cost buys nothing that survives.
#
# Set to 0 there and the ingest endpoints refuse up front with a 503 instead
# of accepting a job that will be killed half way through. The library then
# comes from the seed baked into the image: index on a real machine, run
# scripts/build_seed.py, commit, redeploy.
#
# Default ON -- a laptop and any host with a mounted disk should ingest
# normally, and a flag that quietly disabled the feature everywhere would be
# a worse failure than the one it prevents.
ALLOW_RUNTIME_INGEST = os.environ.get(
    "ALLOW_RUNTIME_INGEST", "1") not in {"0", "false", "no"}

# --- position model -------------------------------------------------------
# word offset is ground truth; pages are a synthetic view over it.
WORDS_PER_PAGE = 250

# --- chunking -------------------------------------------------------------
# 320 rather than 200: a narrative answer needs surrounding context, and the
# screenshot evidence showed 200-word passages cutting off mid-exchange.
CHUNK_WORDS = 320
# 40 rather than 80. Overlap is redundant compute: at 80 every word of the
# book is embedded 1.25 times, and embedding is the CPU high-water mark on a
# fractional-core instance. Halving it cuts ~10% off an ingest for a small
# loss of continuity across chunk seams -- which neighbour expansion in
# retrieval already covers, since adjacent chunks are pulled in together.
#
# Chunking happens at INGEST, so this rebinds nothing already on disk: books
# already in the index keep the boundaries they were built with, and their
# stored matrices stay valid. Only newly ingested books use the new stride.
CHUNK_OVERLAP = 40

# --- embeddings -----------------------------------------------------------
# fastembed/ONNX, in this process. Hermetic and free; the cost is ~190 MB
# resident, which is the memory high-water mark of the app (CHALLENGES.md §6).
#
# The dimension is not cosmetic -- it is the width of every matrix written to
# disk, and vectors.save() rejects a mismatch. Changing the model invalidates
# every stored embedding and requires a full re-index of every book.
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


# Completion tokens are metered alongside the prompt, and on gpt-oss the
# reasoning trace is billed as completion -- so an unbounded max_tokens is an
# unbounded bill for a one-word reply. Size each role for the longest
# LEGITIMATE output it has:
#   answer -> four sentences, with room to reason over the passages first
#   fast   -> a single word: an intent, or GROUNDED/UNGROUNDED
# Measured completions are 143 (answer) and 35-74 (fast), so these are
# several times headroom, not a tight fit. Too small is not a wrong answer
# but an empty one, which surfaces as a provider error.
MAX_COMPLETION_TOKENS = {"answer": 700, "fast": 256}


def model_for(role: str, provider: str) -> str:
    """Resolve the model for a node role.

    Precedence: LLM_MODEL_<ROLE> -> LLM_MODEL -> provider default.
    """
    override = (os.environ.get(f"LLM_MODEL_{role.upper()}")
                or os.environ.get("LLM_MODEL"))
    if override:
        return override
    return _DEFAULT_MODELS.get(provider, {}).get(role, "")

# --- question analysis ----------------------------------------------------
#
# `intent` is a DISPLAY label. It is returned to the UI and shown next to the
# answer; it does not steer retrieval, does not enter any prompt, and does
# not affect routing. Paying an LLM call for it costs ~240 tokens and ~0.6s
# on every question to decorate a badge.
#
# The keyword table in app/agents/analyzer.py already resolves most
# questions for free. Set to 1 to restore the model fallback for the
# remainder; the default trades a slightly less accurate badge for a fifth
# of the per-question token cost.
ANALYZER_USE_MODEL = os.environ.get("ANALYZER_USE_MODEL", "0") not in {"0", "false", "no"}

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
# This is the single biggest line item in a question: at 9,000 chars the
# answer prompt measures ~2,250 tokens, which is ~90% of the whole turn.
# It is also the answer-quality knob -- passages dropped here are passages
# the model cannot cite -- so it is env-tunable rather than tuned down.
# Roughly: tokens_per_question ~= MAX_EVIDENCE_CHARS/4 + 400.
MAX_EVIDENCE_CHARS = int(os.environ.get("MAX_EVIDENCE_CHARS", "9000"))

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
