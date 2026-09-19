"""RestoGuide AI — a domain-specific RAG assistant for restaurant operations.

The two pipelines under this package are strictly isolated: `src.ingestion`
writes the vector store, `src.query` reads it, and neither imports the other.
They communicate only through the persisted ChromaDB collection, whose shape is
fixed by the contract in IMPLEMENTATION.md section 3.
"""

__version__ = "0.1.0"
