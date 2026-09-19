"""Read and validate the curated knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.schema import RecordError, validate_record


class IngestionError(RuntimeError):
    """Raised when the ingestion pipeline cannot proceed."""


def load_records(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load data.json and return validated, normalised records.

    Validation is all-or-nothing: one bad record fails the whole run. A
    partially indexed knowledge base is worse than no index at all, because
    retrieval silently returns the nearest surviving neighbour instead of the
    procedure the user actually asked for.
    """
    data_file = Path(path) if path is not None else config.DATA_FILE

    try:
        # utf-8-sig, not utf-8: Notepad and PowerShell's Set-Content write a BOM,
        # and a plain utf-8 read then rejects otherwise-valid JSON with a message
        # about byte order marks that tells an author nothing useful.
        raw_text = data_file.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise IngestionError(f"data file not found at {data_file}") from exc
    except OSError as exc:
        raise IngestionError(f"could not read {data_file}: {exc}") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise IngestionError(
            f"{data_file} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(payload, list):
        raise IngestionError(
            f"expected a JSON array of records in {data_file}, "
            f"got {type(payload).__name__}"
        )

    if not payload:
        raise IngestionError(f"{data_file} contains no records")

    records: list[dict[str, str]] = []
    for index, raw in enumerate(payload):
        try:
            records.append(validate_record(raw, index))
        except RecordError as exc:
            raise IngestionError(f"{data_file}: {exc}") from exc

    _warn_on_duplicates(records)
    return records


def _warn_on_duplicates(records: list[dict[str, str]]) -> None:
    """Report duplicate questions without failing.

    Duplicates waste an embedding and split retrieval between two identical
    chunks, but they do not corrupt the knowledge base, so they are worth a
    warning rather than an abort.
    """
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        key = record["question"].casefold()
        if key in seen:
            print(
                f"WARNING: records {seen[key]} and {index} have the same question; "
                f"both will be indexed"
            )
        else:
            seen[key] = index
