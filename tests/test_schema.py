"""Phase 1 tests: record validation and the strategy-aware text formatters."""

import pytest

from src.config import (
    STRATEGY_QUESTION_ANSWER,
    STRATEGY_QUESTION_ONLY,
)
from src.schema import (
    RecordError,
    format_context_text,
    format_embedding_text,
    make_record_id,
    validate_record,
)

QUESTION = "What is the recipe for Kenyan Beef Wet Fry?"
ANSWER = "Boil cubed beef until tender, then simmer in a thick tomato gravy."


# --- validate_record ---------------------------------------------------------


def test_valid_record_is_normalised():
    result = validate_record({"question": f"  {QUESTION}  ", "answer": ANSWER}, 0)
    assert result == {"question": QUESTION, "answer": ANSWER}


@pytest.mark.parametrize("field", ["question", "answer"])
def test_missing_field_is_rejected(field):
    record = {"question": QUESTION, "answer": ANSWER}
    del record[field]
    with pytest.raises(RecordError, match=field):
        validate_record(record, 3)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_empty_field_is_rejected(blank):
    with pytest.raises(RecordError, match="empty"):
        validate_record({"question": QUESTION, "answer": blank}, 4)


def test_non_string_field_is_rejected():
    with pytest.raises(RecordError, match="must be a string"):
        validate_record({"question": QUESTION, "answer": 42}, 5)


def test_non_dict_entry_is_rejected():
    with pytest.raises(RecordError, match="expected a JSON object"):
        validate_record(["not", "a", "record"], 6)


def test_error_message_names_the_offending_index():
    with pytest.raises(RecordError, match="record 11"):
        validate_record({"question": QUESTION}, 11)


def test_explicit_valid_category_is_kept():
    result = validate_record(
        {"question": QUESTION, "answer": ANSWER, "category": "kitchen"}, 0
    )
    assert result["category"] == "kitchen"


def test_explicit_invalid_category_is_rejected():
    with pytest.raises(RecordError, match="category"):
        validate_record(
            {"question": QUESTION, "answer": ANSWER, "category": "dessert"}, 0
        )


def test_unknown_keys_are_ignored():
    result = validate_record(
        {"question": QUESTION, "answer": ANSWER, "author": "Patience"}, 0
    )
    assert "author" not in result


# --- IDs ---------------------------------------------------------------------


def test_record_ids_are_zero_padded_and_ordered():
    assert make_record_id(0) == "rec-000"
    assert make_record_id(15) == "rec-015"
    assert sorted([make_record_id(i) for i in range(12)]) == [
        make_record_id(i) for i in range(12)
    ]


# --- formatters --------------------------------------------------------------


def test_context_text_round_trips_both_fields():
    text = format_context_text(QUESTION, ANSWER)
    assert text == f"Question: {QUESTION}\nAnswer: {ANSWER}"
    assert QUESTION in text and ANSWER in text


def test_strategy_a_embeds_the_question_alone():
    """The core guarantee of Strategy A: no answer text, no scaffolding."""
    text = format_embedding_text(QUESTION, ANSWER, STRATEGY_QUESTION_ONLY)
    assert text == QUESTION
    assert ANSWER not in text
    assert "Question:" not in text
    assert "Answer:" not in text


def test_strategy_b_embeds_the_combined_pair():
    text = format_embedding_text(QUESTION, ANSWER, STRATEGY_QUESTION_ANSWER)
    assert text == format_context_text(QUESTION, ANSWER)


def test_the_two_strategies_differ():
    """If these ever coincide, the A/B comparison is measuring nothing."""
    assert format_embedding_text(
        QUESTION, ANSWER, STRATEGY_QUESTION_ONLY
    ) != format_embedding_text(QUESTION, ANSWER, STRATEGY_QUESTION_ANSWER)


def test_unknown_strategy_raises():
    with pytest.raises(RecordError, match="unknown embedding strategy"):
        format_embedding_text(QUESTION, ANSWER, "semantic_vibes")
