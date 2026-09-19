"""CLI entrypoint for the query pipeline.

    python -m src.query.run_query "How do I prep the beef wet fry?"
    python -m src.query.run_query "..." --show-context --top-k 5
"""

from __future__ import annotations

import argparse
import sys

from src.config import ConfigError
from src.query.engine import answer_question
from src.query.retriever import KnowledgeRetriever, QueryError


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_query",
        description="Ask RestoGuide AI a question about restaurant procedures.",
    )
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--top-k", type=int, help="number of chunks to retrieve")
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="print the retrieved chunks and their scores before the answer",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        retriever = KnowledgeRetriever(top_k=args.top_k)

        if args.show_context:
            # Shown without the floor applied, so a near-miss is visible rather
            # than silently discarded -- this is the diagnostic view.
            print(f"strategy: {retriever.embedding_strategy}\n")
            print("RETRIEVED (floor not applied):")
            for chunk in retriever.retrieve(args.question, apply_floor=False):
                mark = "keep" if chunk.score >= retriever.min_relevance else "drop"
                print(
                    f"  [{mark}] {chunk.record_id} | {chunk.category} | "
                    f"score={chunk.score:.4f} | {chunk.question[:70]}"
                )
            print()

        answer = answer_question(args.question, top_k=args.top_k, retriever=retriever)
    except (QueryError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(answer.text)

    if answer.sources:
        cited = ", ".join(
            f"{chunk.record_id} ({chunk.score:.3f})" for chunk in answer.sources
        )
        print(f"\nsources: {cited}")
    print(f"latency: {answer.latency_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
