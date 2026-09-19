"""Build the persisted ChromaDB collection from validated records.

Every run is a full rebuild, so a stale collection can never silently coexist
with new data. The rebuild writes to a temporary directory and swaps it into
place only on success -- a failed embedding run therefore leaves the previous
working index intact rather than destroying it.
"""

from __future__ import annotations

import gc
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config
from src.ingestion.chunker import records_to_documents
from src.ingestion.embeddings import get_ingestion_embeddings
from src.ingestion.loader import IngestionError
from src.schema import format_embedding_text

# Embedding calls are the one network dependency of this pipeline; a transient
# 5xx or rate limit should cost a retry, not the whole knowledge base.
RETRY_DELAYS_SECONDS = (2, 4, 8)


@dataclass(frozen=True)
class IndexReport:
    """Result of an ingestion run."""

    record_count: int
    chunk_count: int
    collection: str
    persist_dir: Path
    embedding_model: str
    strategy: str
    dry_run: bool

    def describe(self) -> str:
        mode = "DRY RUN (no API calls, nothing written)" if self.dry_run else "indexed"
        return (
            f"{mode}: {self.chunk_count} chunks from {self.record_count} records\n"
            f"  strategy   : {self.strategy}\n"
            f"  embeddings : {self.embedding_model}\n"
            f"  collection : {self.collection}\n"
            f"  location   : {self.persist_dir}"
        )


def reset_store(persist_dir: Path) -> None:
    """Delete the persisted store. Full-rebuild semantics."""
    try:
        shutil.rmtree(persist_dir, ignore_errors=False)
    except FileNotFoundError:
        return
    except PermissionError as exc:
        raise IngestionError(
            f"cannot remove {persist_dir} -- close the Streamlit app or any "
            f"process holding the vector store, then re-run"
        ) from exc


def _verify_documents(documents: list[Document], strategy: str) -> None:
    """Guard the contract at write time, not merely in tests."""
    for doc in documents:
        question = doc.metadata.get("question", "")
        answer = doc.metadata.get("answer", "")

        if not answer:
            raise IngestionError(
                f"{doc.metadata.get('record_id')}: answer payload is missing -- "
                f"a chunk without an answer is retrievable but unanswerable"
            )

        expected = format_embedding_text(question, answer, strategy)
        if doc.page_content != expected:
            raise IngestionError(
                f"{doc.metadata.get('record_id')}: embedded text does not match "
                f"strategy {strategy!r}"
            )


def _release_chroma() -> None:
    """Drop Chroma's cached client so the sqlite file handle is closed.

    Chroma caches its system instance process-wide, so simply discarding the
    vector store object is not enough -- on Windows the still-open handle makes
    the staging directory unrenameable.
    """
    SharedSystemClient.clear_system_cache()
    gc.collect()


def _embed_with_retry(
    documents: list[Document], ids: list[str], persist_dir: Path, collection: str, strategy: str
) -> None:
    """Write the collection, retrying transient API failures with backoff."""
    embeddings = get_ingestion_embeddings()
    last_error: Exception | None = None

    for attempt, delay in enumerate((*RETRY_DELAYS_SECONDS, None), start=1):
        try:
            store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                ids=ids,
                collection_name=collection,
                persist_directory=str(persist_dir),
                collection_metadata={
                    # Chroma defaults to l2; the contract specifies cosine, and
                    # the query side's relevance floor is calibrated for it.
                    "hnsw:space": "cosine",
                    "embedding_strategy": strategy,
                    "embedding_model": config.EMBEDDING_MODEL,
                },
            )
            del store
            _release_chroma()
            return
        except Exception as exc:  # noqa: BLE001 - provider raises a wide range
            last_error = exc
            if delay is None:
                break
            print(
                f"  embedding attempt {attempt} failed ({type(exc).__name__}); "
                f"retrying in {delay}s"
            )
            time.sleep(delay)

    raise IngestionError(
        f"embedding failed after {len(RETRY_DELAYS_SECONDS) + 1} attempts: {last_error}"
    ) from last_error


def _swap_into_place(staging_dir: Path, persist_dir: Path) -> None:
    """Replace the live store with the freshly built one.

    Chroma may still hold file handles briefly after the write, so the rename
    is retried before giving up -- this is the common Windows failure mode.
    """
    _release_chroma()

    for delay in (0, 0.5, 1.0, 2.0):
        if delay:
            time.sleep(delay)
        try:
            if persist_dir.exists():
                shutil.rmtree(persist_dir)
            staging_dir.rename(persist_dir)
            return
        except PermissionError:
            continue

    raise IngestionError(
        f"built the index at {staging_dir} but could not move it to {persist_dir} -- "
        f"close any process using the vector store and re-run"
    )


def build_index(
    records: list[dict[str, str]],
    *,
    strategy: str | None = None,
    dry_run: bool = False,
    persist_dir: Path | None = None,
    collection: str | None = None,
) -> IndexReport:
    """Chunk, embed, and persist the knowledge base.

    With `dry_run=True` no embedding client is constructed at all, so the
    preview path is provably free of API calls and works without a key.
    """
    strategy = strategy or config.EMBEDDING_STRATEGY
    persist_dir = Path(persist_dir) if persist_dir else config.CHROMA_PERSIST_DIR
    collection = collection or config.CHROMA_COLLECTION

    documents = records_to_documents(records, strategy)
    _verify_documents(documents, strategy)

    if len(documents) != len(records):
        raise IngestionError(
            f"chunk count {len(documents)} does not equal record count "
            f"{len(records)} -- record-based chunking is broken"
        )

    report = IndexReport(
        record_count=len(records),
        chunk_count=len(documents),
        collection=collection,
        persist_dir=persist_dir,
        embedding_model=config.EMBEDDING_MODEL,
        strategy=strategy,
        dry_run=dry_run,
    )

    if dry_run:
        return report

    ids = [doc.metadata["record_id"] for doc in documents]
    staging_dir = persist_dir.with_name(f"{persist_dir.name}.building")
    reset_store(staging_dir)

    _embed_with_retry(documents, ids, staging_dir, collection, strategy)
    _swap_into_place(staging_dir, persist_dir)

    return report
