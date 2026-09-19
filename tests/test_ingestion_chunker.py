"""Phase 2 tests: record-based chunking and the strategy contract."""

import pytest

from src.config import STRATEGY_QUESTION_ANSWER, STRATEGY_QUESTION_ONLY
from src.ingestion.chunker import record_to_document, records_to_documents
from src.ingestion.loader import load_records
from src.schema import format_context_text

RECORD = {
    "question": "What is the ratio of maize to beans for Githeri?",
    "answer": "Two cups of maize to three cups of beans.",
}

METADATA_KEYS = {"record_id", "category", "source", "question", "answer"}


def test_one_record_becomes_exactly_one_chunk():
    records = load_records()
    documents = records_to_documents(records, STRATEGY_QUESTION_ONLY)
    assert len(documents) == len(records) == 16


def test_ids_follow_file_order():
    documents = records_to_documents(load_records(), STRATEGY_QUESTION_ONLY)
    assert [doc.metadata["record_id"] for doc in documents[:3]] == [
        "rec-000",
        "rec-001",
        "rec-002",
    ]
    assert documents[15].metadata["record_id"] == "rec-015"


def test_all_metadata_keys_present_and_non_empty():
    for doc in records_to_documents(load_records(), STRATEGY_QUESTION_ONLY):
        assert set(doc.metadata) == METADATA_KEYS
        for key, value in doc.metadata.items():
            assert isinstance(value, str) and value.strip(), key


def test_strategy_a_embeds_the_question_only():
    doc = record_to_document(RECORD, 0, STRATEGY_QUESTION_ONLY)
    assert doc.page_content == RECORD["question"]
    assert RECORD["answer"] not in doc.page_content


def test_strategy_b_embeds_the_pair():
    doc = record_to_document(RECORD, 0, STRATEGY_QUESTION_ANSWER)
    assert doc.page_content == format_context_text(
        RECORD["question"], RECORD["answer"]
    )


@pytest.mark.parametrize(
    "strategy", [STRATEGY_QUESTION_ONLY, STRATEGY_QUESTION_ANSWER]
)
def test_answer_payload_is_stored_under_both_strategies(strategy):
    """The query pipeline reads the answer from metadata, whatever the strategy."""
    doc = record_to_document(RECORD, 0, strategy)
    assert doc.metadata["answer"] == RECORD["answer"]
    assert doc.metadata["question"] == RECORD["question"]


def test_explicit_category_overrides_the_keyword_rules():
    record = {**RECORD, "category": "admin"}
    assert record_to_document(record, 0, STRATEGY_QUESTION_ONLY).metadata["category"] == "admin"


def test_no_length_based_splitting_of_the_longest_record():
    """The longest record must survive whole -- splitting would sever a procedure."""
    records = load_records()
    longest = max(records, key=lambda r: len(r["answer"]))
    index = records.index(longest)
    doc = record_to_document(longest, index, STRATEGY_QUESTION_ANSWER)
    assert longest["answer"] in doc.page_content
    assert doc.metadata["answer"] == longest["answer"]
