"""Record-based chunking: one JSON record becomes exactly one Document.

No text splitter is used anywhere in this module, deliberately. Each record is
already one self-contained procedure; splitting by character or token count
would sever a recipe or a safety checklist mid-step and leave the retriever
with fragments that are meaningless on their own.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.ingestion.categorizer import classify_category
from src.schema import format_embedding_text, make_record_id


def record_to_document(
    record: dict[str, str], index: int, strategy: str, source: str = "data.json"
) -> Document:
    """Convert one validated record into a Document ready for embedding.

    `page_content` is what Chroma vectorises, so it holds whatever the active
    strategy embeds. The answer is *always* stored in metadata regardless of
    strategy, which is what lets the query pipeline read it from one place and
    stay strategy-agnostic.
    """
    question = record["question"]
    answer = record["answer"]
    category = record.get("category") or classify_category(question, answer)

    return Document(
        page_content=format_embedding_text(question, answer, strategy),
        metadata={
            "record_id": make_record_id(index),
            "category": category,
            "source": source,
            "question": question,
            "answer": answer,
        },
    )


def records_to_documents(
    records: list[dict[str, str]], strategy: str, source: str = "data.json"
) -> list[Document]:
    """Map records to Documents, preserving the one-to-one invariant."""
    documents = [
        record_to_document(record, index, strategy, source)
        for index, record in enumerate(records)
    ]

    if len(documents) != len(records):
        raise AssertionError(
            f"record-based chunking violated: {len(records)} records produced "
            f"{len(documents)} chunks"
        )

    return documents
