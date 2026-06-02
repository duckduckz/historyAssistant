from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from pprint import pprint

from elasticsearch import ElasticsearchWarning

warnings.filterwarnings("ignore", category=ElasticsearchWarning)


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "configs").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not find repo root")


REPO_ROOT = find_repo_root(Path.cwd())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval.eval_nurse_benchmark import compare_answer, run_pipeline
from historyAssistant.app.runtime import load_runtime
from historyAssistant.execution import open_es_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark question or one custom question.")
    parser.add_argument("--id", help="Benchmark ID, for example T040")
    parser.add_argument("--question", help="Custom question to run")
    parser.add_argument(
        "--scenario",
        default="simple_lookup",
        help="Expected scenario for custom questions. Ignored when --id is used.",
    )
    parser.add_argument(
        "--benchmark",
        default="data/historyCalls/04_ground_truth/ground_truth_benchmark_nurse_30.json",
        help="Path to benchmark JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.id and not args.question:
        raise SystemExit("Provide --id T040 or --question \"...\"")

    benchmark_path = REPO_ROOT / args.benchmark
    benchmark_records = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_by_id = {item["id"]: item for item in benchmark_records}

    if args.id:
        gt = benchmark_by_id.get(args.id)
        if gt is None:
            raise SystemExit(f"Unknown benchmark id: {args.id}")
        question = gt["question"]
        expected_scenario = gt["scenario"]
    else:
        gt = None
        question = args.question
        expected_scenario = args.scenario

    print("Question:")
    print(question)
    print()
    print("Expected scenario:", expected_scenario)
    if gt:
        print("Expected output type:", gt.get("expected_output_type"))
        print("Expected answer:")
        pprint(gt.get("expected_answer"), sort_dicts=False)
        print()

    bundle, client, examples = load_runtime()
    es = open_es_connection(config_bundle=bundle)
    result = run_pipeline(question, expected_scenario, bundle, client, examples, es)

    match_ok = None
    match_detail = "custom question: no benchmark comparison"
    if gt is not None:
        match_ok, match_detail = compare_answer(result, gt)

    print("Match:", match_ok)
    print("Match detail:", match_detail)
    print("Scenario predicted:", result.get("scenario"))
    print("Scenario match:", result.get("scenario_match"))
    print("Total time (sec):", result.get("total_time_sec"))
    print()

    compact = {
        "validation_valid": result.get("validation_valid"),
        "repairs": result.get("repairs"),
        "timings": result.get("timings"),
        "scalar_answer": result.get("scalar_answer"),
        "record_count": len(result.get("records") or []),
        "where_debug": result.get("where_debug"),
        "comparison_summary": result.get("comparison_summary"),
        "error": result.get("error"),
    }
    print("Compact debug:")
    pprint(compact, sort_dicts=False)
    print()

    records = result.get("records") or []
    if records:
        print("Preview rows:")
        for row in records[:10]:
            pprint(row, sort_dicts=False)
            print("-" * 60)

    planning_entry = result.get("planning_entry") or {}
    validation = planning_entry.get("validation") or {}
    canonical_plan = validation.get("canonical_plan") or result.get("canonical_plan")
    if canonical_plan:
        print("Canonical plan:")
        pprint(canonical_plan, sort_dicts=False)


if __name__ == "__main__":
    main()
