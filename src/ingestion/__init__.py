"""Ingestion pipeline: data.json -> validated records -> chunks -> ChromaDB.

The write side of the system. This package must never import from `src.query`;
the two communicate only through the persisted collection described in
IMPLEMENTATION.md section 3. `tests/test_isolation.py` enforces that.
"""

from src.ingestion.indexer import IndexReport, build_index, reset_store
from src.ingestion.loader import IngestionError, load_records

__all__ = [
    "IndexReport",
    "IngestionError",
    "build_index",
    "load_records",
    "reset_store",
]
