"""Benchmark conversational follow-ups against eval/benchmark_multiturn.json.

    python eval/run_eval_multiturn.py
    python eval/run_eval_multiturn.py --no-condense    # baseline: stateless

Each case replays its `setup` turns through the real pipeline to build genuine
history, then asks the follow-up. Two things are measured: whether the
follow-up now retrieves the record it means, and whether out-of-scope
follow-ups are still refused. The second matters more -- a condenser that
fixes retrieval by dragging every question in-domain has broken the guardrails,
not helped them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.query.engine import answer_question  # noqa: E402
from src.query.retriever import KnowledgeRetriever  # noqa: E402

BENCHMARK_FILE = ROOT / "eval" / "benchmark_multiturn.json"
FOLLOW_UP = "follow_up"
OUT_OF_SCOPE = "out_of_scope_follow_up"

# The query path has no retry of its own, and a transient 504 partway through a
# run would otherwise discard every case before it.
RETRY_DELAYS_SECONDS = (2, 4, 8)


def _with_retry(fn):
    for delay in (*RETRY_DELAYS_SECONDS, None):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - provider raises a wide range
            if delay is None:
                raise
            time.sleep(delay)


@dataclass
class Result:
    case: dict
    searched_for: str = ""
    retrieved: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    answer_text: str = ""
    refused: bool = False
    latency_ms: float = 0.0

    @property
    def is_negative(self) -> bool:
        return self.case["expected_record_id"] is None

    @property
    def hit(self) -> bool:
        return self.case["expected_record_id"] in self.retrieved

    @property
    def passed(self) -> bool:
        if self.is_negative:
            return self.refused
        return self.hit and not self.refused


def run_case(case: dict, retriever: KnowledgeRetriever, condense: bool) -> Result:
    """Replay the setup turns, then ask the follow-up with that history."""
    history: list[tuple[str, str]] = []

    for setup_query in case["setup"]:
        answer = _with_retry(
            lambda q=setup_query: answer_question(
                q, history=history if condense else None, retriever=retriever
            )
        )
        history.append((setup_query, answer.text))

    query = case["query"]
    answer = _with_retry(
        lambda: answer_question(
            query, history=history if condense else None, retriever=retriever
        )
    )

    searched = answer.searched_for or query
    unfiltered = _with_retry(lambda: retriever.retrieve(searched, apply_floor=False))

    return Result(
        case=case,
        searched_for=searched,
        retrieved=[c.record_id for c in unfiltered],
        kept=[c.record_id for c in answer.sources],
        scores=[round(c.score, 4) for c in unfiltered],
        answer_text=answer.text,
        refused=answer.refused,
        latency_ms=answer.latency_ms,
    )


def _print_line(result: Result) -> None:
    case = result.case
    mark = "PASS" if result.passed else "FAIL"

    if result.is_negative:
        detail = "refused" if result.refused else f"ANSWERED from {result.kept}"
    elif result.refused:
        detail = "REFUSED"
    elif result.hit:
        detail = f"top={result.retrieved[0]}"
    else:
        detail = f"MISS got {result.retrieved[0] if result.retrieved else '-'}"

    print(f"  [{mark}] {case['id']:<13} {detail:<26} {result.latency_ms:6.0f}ms")
    print(f"         searched: {result.searched_for[:88]}")


def summarise(results: list[Result]) -> dict:
    positives = [r for r in results if not r.is_negative]
    negatives = [r for r in results if r.is_negative]

    resolved = [r for r in positives if r.passed]
    refused = [r for r in negatives if r.passed]

    latencies = [r.latency_ms for r in results if r.latency_ms]

    summary = {
        "embedding_strategy": config.EMBEDDING_STRATEGY,
        "llm_model": config.LLM_MODEL,
        "floor_mode": config.FLOOR_MODE,
        "min_relevance": config.MIN_RELEVANCE,
        "counts": {"follow_up": len(positives), "out_of_scope": len(negatives)},
        "follow_up_resolved": len(resolved),
        "follow_up_rate": round(len(resolved) / len(positives), 4) if positives else 0.0,
        "refusal_rate": round(len(refused) / len(negatives), 4) if negatives else 0.0,
        "unresolved": [r.case["id"] for r in positives if not r.passed],
        "leaked": [r.case["id"] for r in negatives if not r.passed],
    }

    if latencies:
        summary["latency_ms"] = {
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "max": round(max(latencies), 1),
        }

    return summary


def report(summary: dict) -> bool:
    print("\n" + "=" * 74)
    print(f"multi-turn  strategy={summary['embedding_strategy']}  floor={summary['floor_mode']}")
    print("=" * 74)
    print(
        f"  follow-ups resolved   {summary['follow_up_resolved']:>2}/"
        f"{summary['counts']['follow_up']:<3} {summary['follow_up_rate']:>7.1%}"
    )
    print(
        f"  out-of-scope refused  "
        f"{summary['counts']['out_of_scope'] - len(summary['leaked']):>2}/"
        f"{summary['counts']['out_of_scope']:<3} {summary['refusal_rate']:>7.1%}"
    )

    if summary["unresolved"]:
        print(f"  Unresolved : {', '.join(summary['unresolved'])}")
    if summary["leaked"]:
        print(f"  LEAKED     : {', '.join(summary['leaked'])}")

    if latency := summary.get("latency_ms"):
        print(
            f"  Latency    mean {latency['mean']:.0f}ms  median "
            f"{latency['median']:.0f}ms  max {latency['max']:.0f}ms"
        )

    # Refusal is the hard gate. A follow-up that stays unresolved is a missed
    # convenience; a leaked out-of-scope answer is a fabricated procedure.
    return summary["refusal_rate"] >= 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_eval_multiturn")
    parser.add_argument(
        "--no-condense",
        action="store_true",
        help="send each follow-up standalone, as the stateless pipeline does",
    )
    parser.add_argument("--out", help="path for the results JSON")
    args = parser.parse_args(argv)

    retriever = KnowledgeRetriever()
    cases = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    condense = not args.no_condense

    mode = "condensed" if condense else "stateless baseline"
    print(f"Running {BENCHMARK_FILE.name} ({mode}) against the live store...\n")

    results = [run_case(case, retriever, condense) for case in cases]
    for result in results:
        _print_line(result)

    summary = summarise(results)
    summary["condense"] = condense
    passed = report(summary)

    suffix = "condensed" if condense else "baseline"
    out_path = (
        Path(args.out).resolve()
        if args.out
        else ROOT / "eval" / f"results.multiturn.{suffix}.json"
    )
    payload = {
        "summary": summary,
        "cases": [
            {
                "id": r.case["id"],
                "category": r.case["category"],
                "setup": r.case["setup"],
                "query": r.case["query"],
                "searched_for": r.searched_for,
                "expected": r.case["expected_record_id"],
                "retrieved": r.retrieved,
                "kept": r.kept,
                "scores": r.scores,
                "refused": r.refused,
                "passed": r.passed,
                "answer": r.answer_text,
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  results -> {out_path.relative_to(ROOT)}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
