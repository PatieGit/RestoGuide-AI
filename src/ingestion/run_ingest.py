"""CLI entrypoint for the ingestion pipeline.

    python -m src.ingestion.run_ingest --dry-run
    python -m src.ingestion.run_ingest
    python -m src.ingestion.run_ingest --strategy question_answer
"""

from __future__ import annotations

import argparse
import sys

from src import config
from src.config import ConfigError
from src.ingestion.indexer import build_index
from src.ingestion.loader import IngestionError, load_records


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ingest",
        description="Build the RestoGuide AI vector store from data.json.",
    )
    parser.add_argument("--data", help="path to data.json (default: config.DATA_FILE)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and preview chunks without making any API calls",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="only ingest the first N records (cheap smoke test)",
    )
    parser.add_argument(
        "--strategy",
        choices=config.VALID_STRATEGIES,
        help=f"embedding strategy (default: {config.EMBEDDING_STRATEGY})",
    )
    return parser.parse_args(argv)


def _preview(records: list[dict[str, str]], strategy: str) -> None:
    """Show what would be embedded, with payload shown separately.

    Embedded text and payload are labelled distinctly so a reviewer can see at
    a glance which one the vector is actually built from.
    """
    from src.ingestion.chunker import records_to_documents

    for doc in records_to_documents(records, strategy):
        meta = doc.metadata
        print(f"\n[{meta['record_id']}] category={meta['category']}")
        print(f"  embedded : {doc.page_content[:110]}")
        print(f"  payload  : {meta['answer'][:110]}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    strategy = args.strategy or config.EMBEDDING_STRATEGY

    try:
        records = load_records(args.data)

        if args.limit is not None:
            if args.limit < 1:
                raise IngestionError("--limit must be at least 1")
            records = records[: args.limit]
            print(f"--limit {args.limit}: ingesting the first {len(records)} records")

        if args.dry_run:
            _preview(records, strategy)

        report = build_index(records, strategy=strategy, dry_run=args.dry_run)
    except (IngestionError, ConfigError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\n{report.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
