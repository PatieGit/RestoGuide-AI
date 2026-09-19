"""Phase 5: the persisted store must match the contract in IMPLEMENTATION.md section 3.

These tests read the live collection, so they are marked `live` even though
they make no API call -- they need an index that ingestion actually built.
"""

from __future__ import annotations

import chromadb
import pytest

from src import config
from src.ingestion.loader import load_records
from src.schema import VALID_CATEGORIES, format_embedding_text, make_record_id

METADATA_KEYS = {"record_id", "category", "source", "question", "answer"}

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def collection():
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
        return client.get_collection(config.CHROMA_COLLECTION)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001
        # A damaged store raises a Rust panic, which is not an Exception.
        pytest.skip(f"vector store unavailable ({exc}); run run_ingest first")


@pytest.fixture(scope="module")
def stored(collection):
    return collection.get(include=["documents", "metadatas", "embeddings"])


def test_collection_is_named_as_configured(collection):
    assert collection.name == config.CHROMA_COLLECTION


def test_chunk_count_equals_record_count(collection):
    assert collection.count() == len(load_records())


def test_collection_declares_its_build_settings(collection):
    metadata = collection.metadata or {}
    assert metadata["embedding_strategy"] in config.VALID_STRATEGIES
    assert metadata["embedding_model"] == config.EMBEDDING_MODEL
    assert metadata["hnsw:space"] == "cosine"


def test_ids_follow_the_documented_format(stored):
    expected = {make_record_id(i) for i in range(len(stored["ids"]))}
    assert set(stored["ids"]) == expected


def test_every_item_carries_all_metadata_keys(stored):
    for cid, meta in zip(stored["ids"], stored["metadatas"]):
        assert set(meta) == METADATA_KEYS, cid
        for key, value in meta.items():
            assert isinstance(value, str) and value.strip(), f"{cid}.{key}"


def test_record_id_mirrors_the_chunk_id(stored):
    for cid, meta in zip(stored["ids"], stored["metadatas"]):
        assert meta["record_id"] == cid


def test_categories_are_from_the_documented_set(stored):
    for meta in stored["metadatas"]:
        assert meta["category"] in VALID_CATEGORIES


def test_page_content_matches_the_declared_strategy(collection, stored):
    """Catches a store whose contents disagree with its own declared strategy."""
    strategy = (collection.metadata or {})["embedding_strategy"]
    for cid, doc, meta in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        assert doc == format_embedding_text(meta["question"], meta["answer"], strategy), cid


def test_answer_payload_survives_the_round_trip(stored):
    """The answer must come back byte-identical to data.json."""
    by_question = {r["question"]: r["answer"] for r in load_records()}
    for meta in stored["metadatas"]:
        assert meta["answer"] == by_question[meta["question"]]


def test_embedding_dimensions_are_consistent(stored):
    dimensions = {len(vector) for vector in stored["embeddings"]}
    assert len(dimensions) == 1, f"inconsistent dimensions: {dimensions}"


def test_query_side_can_read_everything_it_needs(stored):
    """Contract guarantee 4: the answer is read from metadata, never parsed."""
    for meta in stored["metadatas"]:
        assert meta["question"] and meta["answer"]
