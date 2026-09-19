"""Shared configuration for RestoGuide AI.

This module holds *declarations only* — model IDs, paths, and tunables read once
from the environment. It deliberately contains no pipeline behaviour, because it
is the one module both `src.ingestion` and `src.query` are allowed to import.
Putting logic here would reintroduce the coupling the package split exists to
prevent.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def _env(name: str, default: str) -> str:
    """Read an environment variable, falling back to the default if unset/blank."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative configured path against the project root."""
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


# --- Credentials -------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# --- Models ------------------------------------------------------------------
# Every reference to a Gemini model in this codebase goes through these two
# constants, so changing a model is a one-line edit.

LLM_MODEL = _env("GEMINI_LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = _env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# temperature=0 makes answers deterministic, which is what makes the Phase 5
# accuracy benchmark meaningful — the same question must give the same answer.
LLM_TEMPERATURE = 0

# --- Storage -----------------------------------------------------------------

DATA_FILE = _resolve(_env("DATA_FILE", "./data/data.json"))
CHROMA_PERSIST_DIR = _resolve(_env("CHROMA_PERSIST_DIR", "./chroma_db"))
CHROMA_COLLECTION = _env("CHROMA_COLLECTION", "restoguide_kb")

# --- Retrieval ---------------------------------------------------------------

TOP_K = int(_env("TOP_K", "3"))

# Guardrail 1. Under FLOOR_RELATIVE this gates the top hit only.
#
# Phase 3 set this to 0.80 from five queries. The 30-pair benchmark then showed
# the bands genuinely overlap -- the weakest legitimate match scored 0.7884
# while the strongest out-of-scope match scored 0.7984 -- so no threshold
# separates them cleanly and 0.80 cleared the negatives by only 0.0016. At 0.75
# the relative floor answers every legitimate question and leaves the five
# borderline out-of-scope queries to the prompt guardrail, which refused all of
# them. The original 0.35 never fired at all, leaving the guardrail decorative.
#
# Strategy-dependent: a threshold calibrated under question_only is not valid
# under question_answer and must be re-tuned after a switch.
MIN_RELEVANCE = float(_env("MIN_RELEVANCE", "0.75"))

# How the floor is applied:
#   flat     - every chunk must clear MIN_RELEVANCE on its own.
#   relative - MIN_RELEVANCE gates the top hit only; the rest must fall within
#              RELATIVE_MARGIN of it. A raw score means little alone, but a
#              score relative to the best available match means a lot.
FLOOR_FLAT = "flat"
FLOOR_RELATIVE = "relative"
VALID_FLOOR_MODES = (FLOOR_FLAT, FLOOR_RELATIVE)

# Chosen in Phase 5 on the 30-pair benchmark. Relative at gate 0.75 was the only
# configuration that answered all 24 legitimate questions while still refusing
# all 6 out-of-scope ones, and it sends ~1.5 chunks to the model instead of the
# flat floor's ~2.5, so the context carries less weak padding.
FLOOR_MODE = _env("FLOOR_MODE", FLOOR_RELATIVE)
RELATIVE_MARGIN = float(_env("RELATIVE_MARGIN", "0.10"))

if FLOOR_MODE not in VALID_FLOOR_MODES:
    raise ConfigError(
        f"FLOOR_MODE={FLOOR_MODE!r} is not valid. "
        f"Expected one of: {', '.join(VALID_FLOOR_MODES)}"
    )

# --- Embedding strategy ------------------------------------------------------

STRATEGY_QUESTION_ONLY = "question_only"
STRATEGY_QUESTION_ANSWER = "question_answer"
VALID_STRATEGIES = (STRATEGY_QUESTION_ONLY, STRATEGY_QUESTION_ANSWER)

EMBEDDING_STRATEGY = _env("EMBEDDING_STRATEGY", STRATEGY_QUESTION_ONLY)

# Validated at import: a typo must never fall back to a default silently, or we
# would index under one scheme while the benchmark report claims another.
if EMBEDDING_STRATEGY not in VALID_STRATEGIES:
    raise ConfigError(
        f"EMBEDDING_STRATEGY={EMBEDDING_STRATEGY!r} is not valid. "
        f"Expected one of: {', '.join(VALID_STRATEGIES)}"
    )


def require_api_key() -> str:
    """Return the Google API key, or explain how to set it."""
    if not GOOGLE_API_KEY:
        raise ConfigError(
            "GOOGLE_API_KEY not set — copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )
    return GOOGLE_API_KEY


def set_strategy(name: str, env_file: Path | None = None) -> None:
    """Persist the active embedding strategy to .env.

    Used by the automatic Strategy A -> B fallback so the switch survives the
    process that made it. Note this rewrites the file but does not mutate the
    already-imported EMBEDDING_STRATEGY constant; the switch requires a full
    re-ingest in a fresh process anyway.
    """
    if name not in VALID_STRATEGIES:
        raise ConfigError(
            f"Cannot set unknown strategy {name!r}. "
            f"Expected one of: {', '.join(VALID_STRATEGIES)}"
        )

    target = env_file or ENV_FILE
    if not target.exists():
        raise ConfigError(f"Cannot set strategy: {target} does not exist")

    lines = target.read_text(encoding="utf-8").splitlines()
    key = "EMBEDDING_STRATEGY"
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={name}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={name}")

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary() -> str:
    """One-line description of the active configuration, for CLI output."""
    return (
        f"llm={LLM_MODEL} embeddings={EMBEDDING_MODEL} "
        f"strategy={EMBEDDING_STRATEGY} top_k={TOP_K} "
        f"collection={CHROMA_COLLECTION}"
    )
