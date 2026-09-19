"""Benchmark the retrieval and refusal behaviour of RestoGuide AI.

    python eval/run_eval.py
    python eval/run_eval.py --floor-mode relative
    python eval/run_eval.py --no-generate        # retrieval only, no API generation

Reports Hit Rate @ k=3 on in-domain queries and the refusal rate on
out-of-scope queries, against the targets in IMPLEMENTATION.md (85% and 100%).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.query.engine import answer_question  # noqa: E402
from src.query.prompts import NO_ANSWER_TEXT  # noqa: E402
from src.query.retriever import KnowledgeRetriever  # noqa: E402

BENCHMARK_FILE = ROOT / "eval" / "benchmark.json"
HIT_RATE_TARGET = 0.85
REFUSAL_TARGET = 1.0

IN_DOMAIN = ("direct", "paraphrased")


@dataclass
class Result:
    case: dict
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
        """Expected record present in the top-k, before the floor is applied."""
        return self.case["expected_record_id"] in self.retrieved

    @property
    def survived_floor(self) -> bool:
        """Expected record still present after the floor -- what reaches the model."""
        return self.case["expected_record_id"] in self.kept


def evaluate(retriever: KnowledgeRetriever, generate: bool) -> list[Result]:
    cases = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    results: list[Result] = []

    for case in cases:
        query = case["query"]
        unfiltered = retriever.retrieve(query, apply_floor=False)
        kept = retriever.retrieve(query)

        result = Result(
            case=case,
            retrieved=[c.record_id for c in unfiltered],
            kept=[c.record_id for c in kept],
            scores=[round(c.score, 4) for c in unfiltered],
        )

        if generate:
            answer = answer_question(query, retriever=retriever)
            result.answer_text = answer.text
            result.refused = answer.refused
            result.latency_ms = answer.latency_ms

        results.append(result)
        _print_line(result, generate)

    return results


def _print_line(result: Result, generate: bool) -> None:
    case = result.case
    if result.is_negative:
        mark = "PASS" if (result.refused or not result.kept) else "FAIL"
        detail = "refused" if (result.refused or not result.kept) else "ANSWERED"
    else:
        mark = "PASS" if result.hit else "FAIL"
        detail = f"top={result.retrieved[0] if result.retrieved else '-'}"
        if result.hit and not result.survived_floor:
            mark, detail = "WARN", detail + " (dropped by floor)"

    latency = f" {result.latency_ms:6.0f}ms" if generate else ""
    print(f"  [{mark}] {case['id']:<11} {detail:<28}{latency} {case['query'][:52]}")


def summarise(results: list[Result], generate: bool) -> dict:
    in_domain = [r for r in results if not r.is_negative]
    negatives = [r for r in results if r.is_negative]

    hits = [r for r in in_domain if r.hit]
    survived = [r for r in in_domain if r.survived_floor]
    refusals = [r for r in negatives if r.refused or not r.kept]

    hit_rate = len(hits) / len(in_domain) if in_domain else 0.0
    floor_rate = len(survived) / len(in_domain) if in_domain else 0.0
    refusal_rate = len(refusals) / len(negatives) if negatives else 0.0

    latencies = [r.latency_ms for r in results if r.latency_ms]

    summary = {
        "embedding_strategy": config.EMBEDDING_STRATEGY,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "floor_mode": config.FLOOR_MODE,
        "min_relevance": config.MIN_RELEVANCE,
        "relative_margin": config.RELATIVE_MARGIN,
        "top_k": config.TOP_K,
        "counts": {
            "in_domain": len(in_domain),
            "out_of_scope": len(negatives),
        },
        "hit_rate_at_3": round(hit_rate, 4),
        "survived_floor_rate": round(floor_rate, 4),
        "refusal_rate": round(refusal_rate, 4),
        "hit_rate_target_met": hit_rate >= HIT_RATE_TARGET,
        "refusal_target_met": refusal_rate >= REFUSAL_TARGET,
        "misses": [r.case["id"] for r in in_domain if not r.hit],
        "dropped_by_floor": [
            r.case["id"] for r in in_domain if r.hit and not r.survived_floor
        ],
        "leaked": [r.case["id"] for r in negatives if not (r.refused or not r.kept)],
    }

    if latencies:
        summary["latency_ms"] = {
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "max": round(max(latencies), 1),
            "over_3s": sum(1 for value in latencies if value > 3000),
        }

    per_category = {}
    for name in (*IN_DOMAIN, "out_of_scope"):
        group = [r for r in results if r.case["category"] == name]
        if not group:
            continue
        if name == "out_of_scope":
            passed = sum(1 for r in group if r.refused or not r.kept)
        else:
            passed = sum(1 for r in group if r.hit)
        per_category[name] = {
            "passed": passed,
            "total": len(group),
            "rate": round(passed / len(group), 4),
        }
    summary["per_category"] = per_category

    return summary


def report(summary: dict, generate: bool) -> bool:
    print("\n" + "=" * 74)
    print(
        f"strategy={summary['embedding_strategy']}  floor={summary['floor_mode']}"
        f"(min={summary['min_relevance']}, margin={summary['relative_margin']})"
        f"  top_k={summary['top_k']}"
    )
    print("=" * 74)

    for name, stats in summary["per_category"].items():
        print(f"  {name:<14} {stats['passed']:>2}/{stats['total']:<3} {stats['rate']:>7.1%}")

    print("-" * 74)
    hit_flag = "PASS" if summary["hit_rate_target_met"] else "FAIL"
    ref_flag = "PASS" if summary["refusal_target_met"] else "FAIL"
    print(f"  Hit Rate @ k=3   {summary['hit_rate_at_3']:>7.1%}   target 85%   [{hit_flag}]")
    print(f"  Refusal rate     {summary['refusal_rate']:>7.1%}   target 100%  [{ref_flag}]")
    print(f"  Survived floor   {summary['survived_floor_rate']:>7.1%}   (reaches the model)")

    if summary["misses"]:
        print(f"  Retrieval misses : {', '.join(summary['misses'])}")
    if summary["dropped_by_floor"]:
        print(f"  Dropped by floor : {', '.join(summary['dropped_by_floor'])}")
    if summary["leaked"]:
        print(f"  Answered when it should have refused: {', '.join(summary['leaked'])}")

    if latency := summary.get("latency_ms"):
        print(
            f"  Latency          mean {latency['mean']:.0f}ms  median "
            f"{latency['median']:.0f}ms  max {latency['max']:.0f}ms  "
            f"({latency['over_3s']} over 3s)"
        )

    if not summary["hit_rate_target_met"]:
        other = (
            "question_answer"
            if summary["embedding_strategy"] == "question_only"
            else "question_only"
        )
        print(
            f"\n  Hit Rate is below target. Per IMPLEMENTATION.md, rebuild under the\n"
            f"  other strategy and re-measure:\n"
            f"    python -m src.ingestion.run_ingest --strategy {other}\n"
            f"    python eval/run_eval.py"
        )

    return summary["hit_rate_target_met"] and summary["refusal_target_met"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_eval")
    parser.add_argument("--floor-mode", choices=config.VALID_FLOOR_MODES)
    parser.add_argument("--min-relevance", type=float)
    parser.add_argument("--relative-margin", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="retrieval metrics only; skips all LLM calls",
    )
    parser.add_argument("--out", help="path for the results JSON")
    args = parser.parse_args(argv)

    # The report records what actually ran, so overrides land on config.
    if args.floor_mode:
        config.FLOOR_MODE = args.floor_mode
    if args.min_relevance is not None:
        config.MIN_RELEVANCE = args.min_relevance
    if args.relative_margin is not None:
        config.RELATIVE_MARGIN = args.relative_margin
    if args.top_k:
        config.TOP_K = args.top_k

    retriever = KnowledgeRetriever(
        top_k=config.TOP_K,
        min_relevance=config.MIN_RELEVANCE,
        floor_mode=config.FLOOR_MODE,
        relative_margin=config.RELATIVE_MARGIN,
    )

    generate = not args.no_generate
    print(f"Running {BENCHMARK_FILE.name} against the live store...\n")

    results = evaluate(retriever, generate)
    summary = summarise(results, generate)
    passed = report(summary, generate)

    out_path = (
        Path(args.out).resolve()
        if args.out
        else ROOT / "eval" / f"results.{config.EMBEDDING_STRATEGY}.{config.FLOOR_MODE}.json"
    )
    payload = {
        "summary": summary,
        "cases": [
            {
                "id": r.case["id"],
                "category": r.case["category"],
                "query": r.case["query"],
                "expected": r.case["expected_record_id"],
                "retrieved": r.retrieved,
                "kept": r.kept,
                "scores": r.scores,
                "hit": None if r.is_negative else r.hit,
                "refused": r.refused,
                "answer": r.answer_text,
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:
        shown = out_path
    print(f"\n  results -> {shown}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
