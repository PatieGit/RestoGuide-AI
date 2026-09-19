"""Orchestration seam for the query pipeline: retrieve -> prompt -> generate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from src.query.condenser import condense_question
from src.query.generator import generate, get_llm
from src.query.prompts import NO_ANSWER_TEXT, build_prompt
from src.query.retriever import (
    KnowledgeRetriever,
    QueryError,
    RetrievedChunk,
    StoreNotFoundError,
)

EMPTY_QUESTION_TEXT = "Please type a question about the restaurant's procedures."


@dataclass(frozen=True)
class Answer:
    """The result of one query, with the evidence it was based on."""

    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
    latency_ms: float = 0.0
    # What was actually searched for. Differs from the typed question only when
    # a follow-up was condensed; None on single-turn queries.
    searched_for: str | None = None


def answer_question(
    question: str,
    *,
    history: Sequence[tuple[str, str]] | None = None,
    top_k: int | None = None,
    retriever: KnowledgeRetriever | None = None,
    llm=None,
    condense_llm=None,
) -> Answer:
    """Answer a staff question from the knowledge base.

    Two independent guardrails stand between a question and a fabricated
    answer: the relevance floor, which refuses locally without calling the LLM,
    and the system prompt's refusal instruction. A single failure of either one
    should not be enough to produce an invented procedure.

    `history` is a sequence of `(question, answer)` pairs, oldest first. When
    it is present the question is first condensed into a standalone one, so
    everything downstream -- embedding, floor, prompt -- sees the same shape of
    input it was calibrated on. Without history nothing changes and no extra
    API call is made.
    """
    started = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - started) * 1000

    if not question or not question.strip():
        return Answer(text=EMPTY_QUESTION_TEXT, refused=True, latency_ms=elapsed())

    asked = question.strip()
    searched = condense_question(asked, history, llm=condense_llm) if history else asked

    active = retriever or KnowledgeRetriever(top_k=top_k)
    chunks = active.retrieve(searched)

    # Reported only when condensing actually changed the question, so a
    # single-turn answer carries no misleading extra field.
    searched_for = searched if searched != asked else None

    # Guardrail 1: nothing cleared the relevance floor, so refuse locally. This
    # is cheaper and more reliable than trusting the model to refuse.
    if not chunks:
        return Answer(
            text=NO_ANSWER_TEXT,
            refused=True,
            latency_ms=elapsed(),
            searched_for=searched_for,
        )

    # Guardrail 2: the system prompt instructs refusal when the context does
    # not actually contain the answer. The condensed question is what the
    # context was retrieved for, and the model has no conversation of its own,
    # so that is what it must be asked.
    prompt = build_prompt(searched, chunks)
    text = generate(prompt, llm=llm or get_llm())

    return Answer(
        text=text,
        sources=chunks,
        refused=text.strip().rstrip(".").casefold()
        == NO_ANSWER_TEXT.rstrip(".").casefold(),
        latency_ms=elapsed(),
        searched_for=searched_for,
    )


__all__ = [
    "Answer",
    "EMPTY_QUESTION_TEXT",
    "QueryError",
    "StoreNotFoundError",
    "answer_question",
    "condense_question",
]
