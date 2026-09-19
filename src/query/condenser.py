"""Rewrite a conversational follow-up into a standalone question.

The retrieval floor and both guardrails assume a self-contained question: the
question alone is what gets embedded, and MIN_RELEVANCE was calibrated against
questions that stand on their own. A bare follow-up like "what about for fish?"
scores 0.7382 and is refused before the LLM is ever reached.

Condensing restores that assumption rather than weakening it. The rewritten
question then runs through the ordinary retrieve -> floor -> generate path with
nothing else changed, so the calibration stays valid and neither guardrail is
bypassed.
"""

from __future__ import annotations

from typing import Sequence

from src.query.generator import generate
from src.query.prompts import CONDENSE_SYSTEM_PROMPT, build_condense_prompt

# A rewrite is a question, not a paragraph. Anything longer than this, or
# spanning multiple lines, means the model answered or explained instead of
# rewriting -- in which case the original question is the safer input.
MAX_CONDENSED_CHARS = 300


def _looks_like_a_rewrite(text: str) -> bool:
    """Reject output that is clearly an answer rather than a question."""
    if not text:
        return False
    if len(text) > MAX_CONDENSED_CHARS:
        return False
    return "\n" not in text.strip()


def condense_question(
    question: str,
    history: Sequence[tuple[str, str]] | None = None,
    llm=None,
) -> str:
    """Return `question` rewritten to stand on its own.

    With no history there is nothing to resolve, so the question is returned
    untouched and no API call is made -- first turns cost nothing.

    If the rewrite fails or comes back malformed, the original question is
    returned. That degrades to the current stateless behaviour, which is a
    known-safe outcome; raising here would turn a follow-up into an error
    message for an employee mid-shift.
    """
    original = (question or "").strip()
    if not original or not history:
        return original

    prompt = build_condense_prompt(original, history)

    try:
        rewritten = generate(prompt, llm=llm, system=CONDENSE_SYSTEM_PROMPT)
    except Exception:  # noqa: BLE001 - any failure degrades to the raw question
        return original

    rewritten = rewritten.strip().strip('"').strip()

    return rewritten if _looks_like_a_rewrite(rewritten) else original
