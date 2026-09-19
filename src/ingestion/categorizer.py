"""Deterministic category assignment for knowledge base records.

data.json carries no category field, but the retrieval UI cites one and the
proposal calls for metadata tags. Keyword rules keep this free, offline, and
reproducible: the same record always lands in the same category, so an ingest
run can be re-executed without the index shifting underneath the benchmark.
"""

from __future__ import annotations

import re

from src.schema import DEFAULT_CATEGORY

# Order matters: the first matching rule wins. Safety leads deliberately -- an
# allergy or hygiene procedure must be tagged `safety` even when it also
# mentions the kitchen or a guest, because that tag is what surfaces the
# strictest handling downstream.
CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "safety",
        (
            "allergy",
            "allergen",
            "hygiene",
            "sanitize",
            "sanitise",
            "contamination",
            "handwashing",
            "hand washing",
            "wash hands",
            "food safety",
            "spill",
            "caution",
            "temperature displays",
            "cross-contamination",
        ),
    ),
    (
        "kitchen",
        (
            "recipe",
            "prep",
            "cook",
            "boil",
            "saute",
            "sauté",
            "simmer",
            "kitchen",
            "equipment",
            "chiller",
            "ingredient",
            "delivery",
            "deliveries",
            "inventory",
            "microwave",
            "oven",
            "ugali",
            "githeri",
            "beef wet fry",
        ),
    ),
    (
        "front_of_house",
        (
            "guest",
            "server",
            "greet",
            "order",
            "bill",
            "table",
            "complaint",
            "customer",
            "reservation",
            "dining floor",
            "menu",
        ),
    ),
    (
        "admin",
        (
            "cash",
            "register",
            "float",
            "m-pesa",
            "mpesa",
            "paybill",
            "shift report",
            "sign-off",
            "opening routine",
            "closing routine",
            "invoice",
            "reconcile",
        ),
    ),
)


def _compile(keywords: tuple[str, ...]) -> re.Pattern[str]:
    """Match each keyword at a word boundary, allowing suffixes.

    Substring matching is not good enough here: it fires `bill` inside
    `paybill` and `table` inside `vegetables`, which silently misfiles records.
    A leading word boundary plus a trailing `\\w*` matches `cook`/`cooking` and
    `table`/`tables` while refusing to match mid-word.
    """
    alternatives = "|".join(re.escape(keyword) for keyword in keywords)
    return re.compile(rf"\b(?:{alternatives})\w*", re.IGNORECASE)


COMPILED_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (category, _compile(keywords)) for category, keywords in CATEGORY_RULES
)


def classify_category(question: str, answer: str) -> str:
    """Assign a category from the record text.

    Matching is case-insensitive over question and answer combined. Returns
    `general` when no rule fires, which is a legitimate outcome rather than a
    failure -- an untagged record is still fully retrievable.
    """
    haystack = f"{question} {answer}"

    for category, pattern in COMPILED_RULES:
        if pattern.search(haystack):
            return category

    return DEFAULT_CATEGORY
