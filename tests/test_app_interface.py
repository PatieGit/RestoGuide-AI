"""Phase 4 tests: the interface layer stays a pure consumer of src.query."""

import ast
from pathlib import Path

import app
from src.query.prompts import NO_ANSWER_TEXT, SYSTEM_PROMPT

APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


def _imported_modules(source: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_ui_never_imports_the_ingestion_pipeline():
    """The UI structurally cannot write to the index."""
    assert not any(
        module.startswith("src.ingestion") for module in _imported_modules(APP_SOURCE)
    )


def test_ui_consumes_the_query_pipeline():
    assert any(
        module.startswith("src.query") for module in _imported_modules(APP_SOURCE)
    )


def test_ui_holds_no_prompt_logic():
    """Prompt wording lives in src/query/prompts.py, not in the interface."""
    assert "CONTEXT FROM THE MANUAL" not in APP_SOURCE
    assert "Answer using only the context" not in APP_SOURCE


def test_quick_actions_map_to_real_manual_questions():
    from src.ingestion.loader import load_records  # test-only, not imported by app

    stored = {record["question"] for record in load_records()}
    assert len(app.QUICK_ACTIONS) == 5
    for question in app.QUICK_ACTIONS.values():
        assert question in stored


def test_layout_reflows_for_phone_width():
    assert "@media (max-width: 640px)" in app.CSS
    assert "max-width: 46rem" in app.CSS


def test_ui_refusal_wording_comes_from_the_prompt_module():
    """One refusal sentence, defined once: NO_ANSWER_TEXT.

    The UI used to hardcode its own phrasing, so the model's refusal and the
    screen's refusal could drift apart silently.
    """
    assert "NO_ANSWER_TEXT" in APP_SOURCE
    assert "couldn't find this in the restaurant's manual" not in APP_SOURCE
    assert NO_ANSWER_TEXT not in APP_SOURCE.replace("NO_ANSWER_TEXT", "")


def test_prompt_requires_numbered_steps_for_procedures():
    """Adaptive tone must not cost procedures their step structure."""
    lowered = SYSTEM_PROMPT.casefold()
    assert "sequence of actions always gets a numbered list" in lowered
    assert "one or two natural sentences" in lowered


def test_prompt_keeps_step_lists_conversational():
    """A bare "1." with no lead-in reads like a form, not a colleague.

    Tightening rule 2 against false refusals once cost the answers their opening
    line, because "reorganise the prose into steps" read as a formatting order.
    """
    lowered = SYSTEM_PROMPT.casefold()
    assert "one short spoken line" in lowered
    assert "not filling in a form" in lowered
