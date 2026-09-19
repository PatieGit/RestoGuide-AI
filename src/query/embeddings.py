"""Query-side embedding client.

Deliberately duplicated from `src.ingestion.embeddings` rather than shared: the
two pipelines share the *configuration* (`config.EMBEDDING_MODEL`), not a code
path. They must resolve to the same model -- a query embedded by a different
model than the index lands in a different vector space, and the search then
returns confident-looking nonsense with no error to warn you. A drift test in
`tests/test_query_retriever.py` asserts the pairing.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src import config


def qualified_model_name(model: str) -> str:
    """Normalise a bare model id to the `models/<id>` form the SDK expects."""
    return model if "/" in model else f"models/{model}"


def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Build the embedding client used to embed an incoming user question."""
    return GoogleGenerativeAIEmbeddings(
        model=qualified_model_name(config.EMBEDDING_MODEL),
        google_api_key=config.require_api_key(),
    )
