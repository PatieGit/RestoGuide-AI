"""Phase 2 tests: deterministic category assignment."""

import pytest

from src.ingestion.categorizer import classify_category
from src.ingestion.loader import load_records
from src.schema import VALID_CATEGORIES


@pytest.mark.parametrize(
    "question,answer,expected",
    [
        (
            "What steps must be followed when a customer informs staff of a severe food allergy?",
            "Inform the kitchen manager immediately and sanitize all prep surfaces.",
            "safety",
        ),
        (
            "What is the recipe and preparation guide for Kenyan Beef Wet Fry?",
            "Boil cubed beef until tender, then simmer in a thick tomato gravy.",
            "kitchen",
        ),
        (
            "What is the standard procedure for greeting guests as they enter?",
            "Guests should be greeted warmly within 30 seconds of arrival.",
            "front_of_house",
        ),
        (
            "How should staff handle cash and closing register balancing?",
            "Count the physical cash float and reconcile M-PESA paybill logs.",
            "admin",
        ),
        ("What is the wifi password?", "It is on the noticeboard.", "general"),
    ],
)
def test_known_records_land_in_the_right_category(question, answer, expected):
    assert classify_category(question, answer) == expected


def test_classification_is_deterministic():
    """Re-running ingest must not shift the index under the benchmark."""
    args = ("How do I sanitize the prep surface?", "Use the sanitiser spray.")
    assert len({classify_category(*args) for _ in range(50)}) == 1


def test_matching_is_case_insensitive():
    assert classify_category("SEVERE ALLERGY PROCEDURE", "INFORM THE MANAGER") == "safety"


def test_safety_wins_over_kitchen():
    """A hygiene rule that also mentions the kitchen must still be `safety`."""
    assert (
        classify_category(
            "What are the daily food safety guidelines in the kitchen?",
            "Keep cold food below 4C and avoid cross-contamination when you cook.",
        )
        == "safety"
    )


def test_every_real_record_gets_a_valid_category():
    for record in load_records():
        assert classify_category(record["question"], record["answer"]) in VALID_CATEGORIES
