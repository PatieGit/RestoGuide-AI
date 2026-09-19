"""Read-only access to the persisted knowledge base.

This module never creates, writes, updates, or deletes anything in the store.
If the collection is missing, that is a setup error to report -- not something
to paper over by building an empty one, which would turn a clear failure into a
system that silently answers nothing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma

from src import config
from src.query.embeddings import get_query_embeddings
from src.schema import format_context_text


class QueryError(RuntimeError):
    """Raised when the query pipeline cannot produce an answer."""


class StoreNotFoundError(QueryError):
    """Raised when the vector store has not been built yet."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved record, reassembled from stored metadata."""

    record_id: str
    category: str
    question: str
    answer: str
    score: float

    @property
    def context_text(self) -> str:
        """The `Question: ... / Answer: ...` block the LLM reads."""
        return format_context_text(self.question, self.answer)


class KnowledgeRetriever:
    """Opens the existing collection and runs similarity search against it."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection: str | None = None,
        top_k: int | None = None,
        min_relevance: float | None = None,
        floor_mode: str | None = None,
        relative_margin: float | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else config.CHROMA_PERSIST_DIR
        self.collection = collection or config.CHROMA_COLLECTION
        self.top_k = top_k if top_k is not None else config.TOP_K
        self.min_relevance = (
            min_relevance if min_relevance is not None else config.MIN_RELEVANCE
        )
        self.floor_mode = floor_mode or config.FLOOR_MODE
        self.relative_margin = (
            relative_margin if relative_margin is not None else config.RELATIVE_MARGIN
        )

        if not self.persist_dir.exists():
            raise StoreNotFoundError(
                f"Vector store not found at {self.persist_dir} -- run: "
                f"python -m src.ingestion.run_ingest"
            )

        try:
            self._store = Chroma(
                collection_name=self.collection,
                embedding_function=get_query_embeddings(),
                persist_directory=str(self.persist_dir),
            )
            count = self._store._collection.count()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001
            # Deliberately broader than Exception: a store damaged by an
            # ungraceful shutdown surfaces as a Rust panic (pyo3's
            # PanicException derives from BaseException), which would otherwise
            # escape as an unreadable traceback in the UI. Since every ingest
            # is a full rebuild, re-running it is always the correct fix.
            raise StoreNotFoundError(
                f"could not open collection {self.collection!r} at "
                f"{self.persist_dir}. The store may be damaged -- this happens "
                f"if the app is force-killed while running. Rebuild it with: "
                f"python -m src.ingestion.run_ingest ({type(exc).__name__}: {exc})"
            ) from exc

        if count == 0:
            raise StoreNotFoundError(
                f"collection {self.collection!r} is empty -- run: "
                f"python -m src.ingestion.run_ingest"
            )

    @property
    def embedding_strategy(self) -> str:
        """The strategy the store was built with, as it declares itself."""
        metadata = self._store._collection.metadata or {}
        return metadata.get("embedding_strategy", "unknown")

    def retrieve(self, query: str, *, apply_floor: bool = True) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query, reassembled from metadata.

        The match is made against `page_content` -- under the default strategy,
        the stored *question* -- while the answer comes back as payload. Chunks
        below the relevance floor are dropped, which is the first of the two
        hallucination guardrails.
        """
        # An exact question match can score fractionally above 1.0 (cosine
        # distance dips just below zero through float error), which makes
        # LangChain warn and dump every retrieved document to the console. The
        # overshoot is meaningless, so silence that one warning and clamp.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Relevance scores must be between 0 and 1"
            )
            hits = self._store.similarity_search_with_relevance_scores(
                query, k=self.top_k
            )

        chunks: list[RetrievedChunk] = []
        for document, raw_score in hits:
            score = min(1.0, max(0.0, float(raw_score)))
            meta = document.metadata
            question = meta.get("question")
            answer = meta.get("answer")

            if not question or not answer:
                raise QueryError(
                    f"chunk {meta.get('record_id')} is missing its question/answer "
                    f"payload -- the vector store is stale or was built by an "
                    f"incompatible ingestion run. Re-run: "
                    f"python -m src.ingestion.run_ingest"
                )

            chunks.append(
                RetrievedChunk(
                    record_id=meta.get("record_id", "unknown"),
                    category=meta.get("category", "general"),
                    question=question,
                    answer=answer,
                    score=score,
                )
            )

        return self._apply_floor(chunks) if apply_floor else chunks

    def _apply_floor(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop chunks too weak to be worth showing the model (guardrail 1)."""
        if not chunks:
            return chunks

        if self.floor_mode == config.FLOOR_RELATIVE:
            # The gate applies to the best match only: if nothing in the store
            # comes close, refuse outright. Otherwise judge the rest against
            # that best match rather than against a fixed bar, so a confident
            # single hit is not padded with weak neighbours, while genuinely
            # tied matches all survive.
            best = max(chunk.score for chunk in chunks)
            if best < self.min_relevance:
                return []
            cutoff = best - self.relative_margin
            return [chunk for chunk in chunks if chunk.score >= cutoff]

        return [chunk for chunk in chunks if chunk.score >= self.min_relevance]
