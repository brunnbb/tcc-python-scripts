"""
Global pipeline configuration.
Paths, model settings, token limits, pricing, etc.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_FOLDER = PROJECT_ROOT / "data" / "in" / "pdfs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "out"
TEXTS_DIR = OUTPUT_DIR / "texts"
OUTPUT_CSV = OUTPUT_DIR / "results.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
ERROR_LOG = OUTPUT_DIR / "errors.log"

# ── Default model ─────────────────────────────────────────────────────────────
MODEL = "gpt-5-mini"

# Per-model pricing (USD per 1 000 tokens)
# Source: https://developers.openai.com/api/docs/pricing

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5-mini": {"input": 0.000250, "output": 0.002000},
    "gpt-5": {"input": 0.001250, "output": 0.010000},
    "gpt-4o": {"input": 0.002500, "output": 0.010000},
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
}

# ── Text extraction limits ────────────────────────────────────────────────────
MAX_CHARS = 48_000  # ~12 000 input tokens per article
MIN_CHARS_THRESHOLD = 200  # below this → PDF has no extractable text

# ── Rate limiting ─────────────────────────────────────────────────────────────
SLEEP_BETWEEN_CALLS = 1.0  # seconds between API calls
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds; doubles on each retry

# ── Fixed output token estimate ───────────────────────────────────────────────
# Output is a structured JSON with answers to the 17 questions.
AVG_OUTPUT_TOKENS = 500
