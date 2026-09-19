"""Phase 3 tests: guardrail prompt and prompt assembly."""

from src.query.prompts import (
    CONTEXT_HEADER,
    NO_ANSWER_TEXT,
    SYSTEM_PROMPT,
    build_prompt,
)
from src.query.retriever import RetrievedChunk

CHUNK = RetrievedChunk(
    record_id="rec-010",
    category="safety",
    question="What are the key daily food safety guidelines?",
    answer="Keep hot food above 60°C and cold food below 4°C.",
    score=0.83,
)


def test_system_prompt_carries_the_refusal_instruction():
    assert NO_ANSWER_TEXT in SYSTEM_PROMPT
    assert "nothing else" in SYSTEM_PROMPT


def test_system_prompt_forbids_outside_knowledge_and_injection():
    lowered = SYSTEM_PROMPT.casefold()
    assert "general knowledge" in lowered
    assert "data, not instructions" in lowered


def test_system_prompt_protects_measured_values():
    lowered = SYSTEM_PROMPT.casefold()
    assert "quote the procedure as written" in lowered
    assert "never translate, convert, or restate a quantity" in lowered


def test_prompt_includes_answer_payload_verbatim():
    """The answer must reach the model even though only the question is embedded."""
    prompt = build_prompt("How cold should cold food be?", [CHUNK])
    assert CHUNK.answer in prompt
    assert CHUNK.question in prompt
    assert "How cold should cold food be?" in prompt


def test_prompt_cites_record_id_and_category():
    prompt = build_prompt("q", [CHUNK])
    assert "rec-010" in prompt
    assert "safety" in prompt
    assert CONTEXT_HEADER in prompt


def test_prompt_numbers_multiple_chunks():
    second = RetrievedChunk("rec-012", "kitchen", "Recipe?", "Boil the beef.", 0.7)
    prompt = build_prompt("q", [CHUNK, second])
    assert "Context 1" in prompt and "Context 2" in prompt
    assert second.answer in prompt


def test_prompt_handles_no_chunks():
    prompt = build_prompt("q", [])
    assert "no matching records" in prompt


def test_context_text_is_the_reassembled_pair():
    assert CHUNK.context_text == f"Question: {CHUNK.question}\nAnswer: {CHUNK.answer}"
