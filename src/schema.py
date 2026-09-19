"""Record schema, validation, and text formatting for RestoGuide AI.

Shared by both pipelines: `src.ingestion` uses it to validate and format records
on the way into the vector store, `src.query` uses `format_context_text` to
reassemble a retrieved chunk for the prompt. Declarations only — no I/O, no
model calls, no store access.
"""

from __future__ import annotations

from typing import Any

from src.config import (
    STRATEGY_QUESTION_ANSWER,
    STRATEGY_QUESTION_ONLY,
    VALID_STRATEGIES,
)

REQUIRED_FIELDS = ("question", "answer")

VALID_CATEGORIES = frozenset(
    {"kitchen", "front_of_house", "safety", "admin", "general"}
)

DEFAULT_CATEGORY = "general"


class RecordError(ValueError):
    """Raised when a data.json record does not satisfy the schema."""


def validate_record(raw: Any, index: int) -> dict[str, str]:
    """Validate one raw record and return it normalised.

    Strips whitespace from both fields. Empty-after-strip is fatal: a record
    with a blank answer is retrievable but unanswerable, which is worse than
    the record being absent entirely.
    """
    if not isinstance(raw, dict):
        raise RecordError(
            f"record {index}: expected a JSON object, got {type(raw).__name__}"
        )

    normalised: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        if field not in raw:
            raise RecordError(f"record {index}: missing required field {field!r}")

        value = raw[field]
        if not isinstance(value, str):
            raise RecordError(
                f"record {index}: field {field!r} must be a string, "
                f"got {type(value).__name__}"
            )

        stripped = value.strip()
        if not stripped:
            raise RecordError(f"record {index}: field {field!r} is empty")

        normalised[field] = stripped

    # An explicit category on the record wins over the ingestion keyword rules.
    category = raw.get("category")
    if category is not None:
        if not isinstance(category, str) or category.strip() not in VALID_CATEGORIES:
            raise RecordError(
                f"record {index}: category {category!r} is not one of "
                f"{sorted(VALID_CATEGORIES)}"
            )
        normalised["category"] = category.strip()

    return normalised


def make_record_id(index: int) -> str:
    """Stable chunk ID derived from the record's position in data.json."""
    return f"rec-{index:03d}"


def format_context_text(question: str, answer: str) -> str:
    """The text the LLM reads. Identical under both embedding strategies."""
    return f"Question: {question}\nAnswer: {answer}"


def format_embedding_text(question: str, answer: str, strategy: str) -> str:
    """The text that gets vectorised — this is the strategy switch.

    Strategy A (`question_only`) embeds the question alone; the answer travels
    as metadata payload. Strategy B (`question_answer`) embeds both together.
    The strategy is passed explicitly rather than read from config so both
    branches stay trivially unit-testable.
    """
    if strategy == STRATEGY_QUESTION_ONLY:
        return question
    if strategy == STRATEGY_QUESTION_ANSWER:
        return format_context_text(question, answer)
    raise RecordError(
        f"unknown embedding strategy {strategy!r}. "
        f"Expected one of: {', '.join(VALID_STRATEGIES)}"
    )
