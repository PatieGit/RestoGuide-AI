"""Query pipeline: user question -> retrieval -> guarded generation -> answer.

The read side of the system. This package must never import from
`src.ingestion`; it consumes only the persisted collection described in
IMPLEMENTATION.md section 3, and never writes to it.
"""

from src.query.condenser import condense_question
from src.query.engine import Answer, QueryError, answer_question
from src.query.retriever import KnowledgeRetriever, RetrievedChunk, StoreNotFoundError

__all__ = [
    "Answer",
    "KnowledgeRetriever",
    "QueryError",
    "RetrievedChunk",
    "StoreNotFoundError",
    "answer_question",
    "condense_question",
]
