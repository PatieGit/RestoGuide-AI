"""Guardrail 2 must not fire on questions the manual actually answers.

The Phase 5 benchmark measures retrieval (Hit Rate @ k=3) and out-of-scope
refusal, so it cannot see a *false* refusal: a question that retrieves its own
record at score 1.0 and is then refused by the model anyway. rec-003 and rec-005
did exactly that -- the system prompt's restrictions stacked up until the model
read "reformat this prose into steps" as forbidden invention, and answered with
the refusal string instead. Rule 2 now says in so many words that formatting is
not invention.

These call the real API, so they are marked `live`.
"""

from __future__ import annotations

import pytest

from src.ingestion.loader import load_records
from src.query.engine import answer_question
from src.schema import make_record_id

pytestmark = pytest.mark.live

CASES = [
    (make_record_id(index), record["question"])
    for index, record in enumerate(load_records())
]


@pytest.mark.parametrize("record_id,question", CASES, ids=[c[0] for c in CASES])
def test_every_knowledge_base_question_is_answered(record_id, question):
    """A verbatim KB question must never come back as a refusal."""
    result = answer_question(question)

    assert not result.refused, (
        f"{record_id} refused its own question despite retrieving "
        f"itself: {result.text!r}"
    )
    assert result.sources, f"{record_id} produced an answer with no sources"
