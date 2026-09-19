"""Phase 2 tests: loading and validating data.json."""

import json

import pytest

from src import config
from src.ingestion.loader import IngestionError, load_records

VALID = [{"question": "Q one?", "answer": "A one."}]


def _write(tmp_path, payload, *, raw=None):
    path = tmp_path / "data.json"
    path.write_text(
        raw if raw is not None else json.dumps(payload), encoding="utf-8"
    )
    return path


def test_loads_the_real_knowledge_base():
    records = load_records()
    assert len(records) == 16
    assert all({"question", "answer"} <= record.keys() for record in records)


def test_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(IngestionError, match="data file not found"):
        load_records(missing)


def test_malformed_json_reports_line_and_column(tmp_path):
    path = _write(tmp_path, None, raw='[{"question": "Q", "answer": "A",}]')
    with pytest.raises(IngestionError, match="line"):
        load_records(path)


def test_top_level_must_be_a_list(tmp_path):
    path = _write(tmp_path, {"question": "Q", "answer": "A"})
    with pytest.raises(IngestionError, match="expected a JSON array"):
        load_records(path)


def test_empty_file_is_rejected(tmp_path):
    path = _write(tmp_path, [])
    with pytest.raises(IngestionError, match="no records"):
        load_records(path)


def test_one_bad_record_fails_the_whole_run(tmp_path):
    """All-or-nothing: a partial index silently returns the wrong neighbour."""
    path = _write(tmp_path, [*VALID, {"question": "Q two?"}])
    with pytest.raises(IngestionError, match="record 1"):
        load_records(path)


def test_records_are_stripped(tmp_path):
    path = _write(tmp_path, [{"question": "  Q?  ", "answer": "\tA.\n"}])
    assert load_records(path) == [{"question": "Q?", "answer": "A."}]


def test_duplicates_warn_but_do_not_fail(tmp_path, capsys):
    path = _write(tmp_path, [*VALID, {"question": "q ONE?", "answer": "A one."}])
    records = load_records(path)
    assert len(records) == 2
    assert "WARNING" in capsys.readouterr().out


def test_default_path_is_the_configured_data_file():
    assert config.DATA_FILE.name == "data.json"
    assert config.DATA_FILE.exists()
