"""Phase 5: the relevance floor (guardrail 1), both modes.

Tested without a store: `_apply_floor` is pure, so the decision logic is
verifiable offline even though calibrating it required the live benchmark.
"""

import pytest

from src import config
from src.query.retriever import KnowledgeRetriever, RetrievedChunk


def chunk(record_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(record_id, "kitchen", "Q?", "A.", score)


def floor(mode: str, min_relevance=0.75, margin=0.10):
    """A retriever with the floor settings only -- __init__ is bypassed."""
    instance = object.__new__(KnowledgeRetriever)
    instance.floor_mode = mode
    instance.min_relevance = min_relevance
    instance.relative_margin = margin
    return instance


# --- flat -------------------------------------------------------------------


def test_flat_drops_each_chunk_on_its_own_merit():
    kept = floor(config.FLOOR_FLAT)._apply_floor(
        [chunk("a", 0.90), chunk("b", 0.76), chunk("c", 0.60)]
    )
    assert [c.record_id for c in kept] == ["a", "b"]


def test_flat_can_drop_everything():
    assert floor(config.FLOOR_FLAT)._apply_floor([chunk("a", 0.5)]) == []


# --- relative ---------------------------------------------------------------


def test_relative_refuses_when_the_best_match_fails_the_gate():
    """Out-of-scope: nothing in the store is close, so refuse outright."""
    kept = floor(config.FLOOR_RELATIVE)._apply_floor(
        [chunk("a", 0.74), chunk("b", 0.73), chunk("c", 0.70)]
    )
    assert kept == []


def test_relative_keeps_only_what_is_close_to_the_best():
    """A confident single hit is not padded with weak neighbours."""
    kept = floor(config.FLOOR_RELATIVE)._apply_floor(
        [chunk("a", 0.99), chunk("b", 0.83), chunk("c", 0.72)]
    )
    assert [c.record_id for c in kept] == ["a"]


def test_relative_keeps_genuinely_tied_matches():
    """A diffuse question legitimately spans several records."""
    kept = floor(config.FLOOR_RELATIVE)._apply_floor(
        [chunk("a", 0.835), chunk("b", 0.831), chunk("c", 0.829)]
    )
    assert len(kept) == 3


def test_relative_admits_the_weak_but_legitimate_top_hit():
    """para-11 from the benchmark: 0.7884, refused by a flat 0.80 floor."""
    kept = floor(config.FLOOR_RELATIVE)._apply_floor([chunk("rec-008", 0.7884)])
    assert [c.record_id for c in kept] == ["rec-008"]


def test_relative_still_rejects_the_strongest_out_of_scope_hit():
    """neg-06 from the benchmark scored 0.7371 on its top hit under this gate."""
    assert floor(config.FLOOR_RELATIVE, min_relevance=0.80)._apply_floor(
        [chunk("rec-000", 0.7984)]
    ) == []


@pytest.mark.parametrize("mode", config.VALID_FLOOR_MODES)
def test_empty_input_is_safe(mode):
    assert floor(mode)._apply_floor([]) == []


# --- shipped configuration --------------------------------------------------


def test_defaults_match_the_phase_5_decision():
    assert config.FLOOR_MODE == config.FLOOR_RELATIVE
    assert config.MIN_RELEVANCE == 0.75
    assert config.RELATIVE_MARGIN == 0.10
