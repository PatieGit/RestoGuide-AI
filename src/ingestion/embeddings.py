"""Ingestion-side embedding client.

Deliberately duplicated by `src.query.embeddings` rather than shared: the two
pipelines share the *configuration* (`config.EMBEDDING_MODEL`), not a code
path. A drift test asserts both factories resolve to the same model, because a
query embedded by a different model than the index lands in a different vector
space and retrieval returns confident nonsense with no error to warn you.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src import config


def qualified_model_name(model: str) -> str:
    """Normalise a bare model id to the `models/<id>` form the SDK expects."""
    return model if "/" in model else f"models/{model}"


def get_ingestion_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Build the embedding client used to index the knowledge base."""
    return GoogleGenerativeAIEmbeddings(
        model=qualified_model_name(config.EMBEDDING_MODEL),
        google_api_key=config.require_api_key(),
    )
