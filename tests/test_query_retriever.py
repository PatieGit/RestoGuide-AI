"""Phase 3 tests: retrieval, guardrails, and the strategy-agnostic contract.

Tests that reach the Gemini API are marked `live`; run `-m "not live"` to skip.
"""

import pytest

from src import config
from src.query.embeddings import get_query_embeddings, qualified_model_name
from src.query.engine import EMPTY_QUESTION_TEXT, Answer, answer_question
from src.query.prompts import NO_ANSWER_TEXT
from src.query.retriever import (
    KnowledgeRetriever,
    QueryError,
    RetrievedChunk,
    StoreNotFoundError,
)


class FakeRetriever:
    """Stands in for the real store so engine logic is testable offline."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.min_relevance = config.MIN_RELEVANCE

    def retrieve(self, query, *, apply_floor=True):
        return list(self.chunks)


CHUNK = RetrievedChunk(
    record_id="rec-015",
    category="kitchen",
    question="What is the ratio of maize to beans for Githeri?",
    answer="Two cups of maize to three cups of beans.",
    score=0.99,
)


# --- offline: engine guardrails ---------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_question_short_circuits(blank):
    """No retrieval, no API call, no cost."""
    answer = answer_question(blank, retriever=FakeRetriever([CHUNK]))
    assert answer.text == EMPTY_QUESTION_TEXT
    assert answer.refused is True
    assert answer.sources == []


def test_no_surviving_chunks_refuses_without_calling_the_llm():
    """Guardrail 1: the relevance floor refuses locally."""

    def explode(*args, **kwargs):
        raise AssertionError("the LLM must not be called when nothing was retrieved")

    answer = answer_question(
        "How do I change a car tire?", retriever=FakeRetriever([]), llm=explode
    )
    assert answer.text == NO_ANSWER_TEXT
    assert answer.refused is True


def test_answer_carries_sources_and_latency():
    answer = Answer(text="ok", sources=[CHUNK], latency_ms=12.5)
    assert answer.sources[0].record_id == "rec-015"
    assert answer.latency_ms == 12.5


# --- offline: configuration pairing -----------------------------------------


def test_query_and_ingestion_embeddings_resolve_to_the_same_model():
    """Drift guard: a mismatch would silently search a different vector space."""
    from src.ingestion import embeddings as ingestion_embeddings

    assert (
        ingestion_embeddings.qualified_model_name(config.EMBEDDING_MODEL)
        == qualified_model_name(config.EMBEDDING_MODEL)
        == "models/gemini-embedding-001"
    )


def test_missing_store_raises_actionable_error(tmp_path):
    with pytest.raises(StoreNotFoundError, match="run_ingest"):
        KnowledgeRetriever(persist_dir=tmp_path / "nope")


def test_store_not_found_is_a_query_error():
    assert issubclass(StoreNotFoundError, QueryError)


# --- live: real store, real API ---------------------------------------------


@pytest.fixture(scope="module")
def retriever():
    try:
        return KnowledgeRetriever()
    except StoreNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.mark.live
def test_store_declares_its_strategy(retriever):
    assert retriever.embedding_strategy in config.VALID_STRATEGIES


@pytest.mark.live
def test_exact_question_retrieves_its_own_record(retriever):
    chunks = retriever.retrieve("What is the ratio of maize to beans for Githeri?")
    assert chunks[0].record_id == "rec-015"
    assert "2 cups of maize" in chunks[0].answer


@pytest.mark.live
def test_answer_body_query_still_finds_its_record(retriever):
    """Strategy A's known blind spot -- tracked, not assumed."""
    chunks = retriever.retrieve("what temperature should cold food be kept at?")
    assert "rec-010" in {chunk.record_id for chunk in chunks}


@pytest.mark.live
def test_retrieved_chunks_carry_full_payload(retriever):
    for chunk in retriever.retrieve("How do I close the kitchen?"):
        assert chunk.question and chunk.answer
        assert chunk.context_text.startswith("Question: ")


@pytest.mark.live
def test_top_k_is_respected(retriever):
    assert len(retriever.retrieve("kitchen", apply_floor=False)) == config.TOP_K


@pytest.mark.live
def test_out_of_domain_question_is_refused():
    answer = answer_question("How do I change a car tire?")
    assert answer.refused is True
    assert answer.text.strip() == NO_ANSWER_TEXT
