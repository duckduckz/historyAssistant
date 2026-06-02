"""
Build a combined dashboard across simple_lookup, aggregation, and
filtering_ranking evaluations.

Usage:
    cd src
    python eval_compare_dashboard.py

Outputs (under 07_outputs/eval_compare/):
    - combined_eval_summary.json
    - combined_eval_dashboard.png
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

YamlDict = Dict[str, Any]

_HERE = Path(__file__).parent
OUT_DIR = _HERE.parent / "07_outputs/eval_compare"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_latest(base_dir: Path, stem: str) -> Path:
    candidates = sorted(base_dir.glob(f"{stem}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No files found for pattern: {base_dir / (stem + '*.json')}")
    # Prefer numeric suffix variants like summary2/results2 when present.
    return candidates[-1]


def _scenario_bundle() -> List[Tuple[str, Path, Path]]:
    simple_dir = _HERE.parent / "07_outputs/eval_simple_lookup"
    agg_dir = _HERE.parent / "07_outputs/eval_aggregation"
    fr_dir = _HERE.parent / "07_outputs/eval_filtering_ranking"
    return [
        (
            "simple_lookup",
            _pick_latest(simple_dir, "simple_lookup_summary"),
            _pick_latest(simple_dir, "simple_lookup_results"),
        ),
        ("aggregation", agg_dir / "aggregation_summary.json", agg_dir / "aggregation_results.json"),
        (
            "filtering_ranking",
            fr_dir / "filtering_ranking_summary.json",
            fr_dir / "filtering_ranking_results.json",
        ),
    ]


def _collect_data() -> Tuple[List[YamlDict], Counter]:
    scenario_rows: List[YamlDict] = []
    root_causes = Counter()

    for scenario, summary_path, results_path in _scenario_bundle():
        summary = _load_json(summary_path)
        results = _load_json(results_path)
        root_causes.update(r.get("root_cause") for r in results if r.get("root_cause"))
        scenario_rows.append(
            {
                "scenario": scenario,
                "summary_path": str(summary_path),
                "results_path": str(results_path),
                "total": summary.get("total", 0),
                "correct": summary.get("correct", 0),
                "wrong": summary.get("wrong", 0),
                "skipped": summary.get("skipped", 0),
                "accuracy_pct": summary.get("accuracy_pct", 0.0),
                "average_latency_sec": summary.get("average_latency_sec", 0.0),
            }
        )

    return scenario_rows, root_causes


def _build_dashboard(rows: List[YamlDict], root_causes: Counter, out_path: Path) -> None:
    scenarios = [r["scenario"] for r in rows]
    accuracies = [r["accuracy_pct"] for r in rows]
    latencies = [r["average_latency_sec"] for r in rows]
    correct = [r["correct"] for r in rows]
    wrong = [r["wrong"] for r in rows]
    total = sum(r["total"] for r in rows)
    total_correct = sum(r["correct"] for r in rows)
    total_wrong = sum(r["wrong"] for r in rows)
    overall_accuracy = round(100 * total_correct / max(total_correct + total_wrong, 1), 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"HistoryAssistant Benchmark Comparison - {overall_accuracy:.1f}% overall accuracy",
        fontsize=15,
        fontweight="bold",
    )

    axes[0, 0].bar(scenarios, accuracies, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title("Accuracy by Scenario")
    axes[0, 0].set_ylabel("%")

    axes[0, 1].bar(scenarios, latencies, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0, 1].set_title("Average Latency by Scenario")
    axes[0, 1].set_ylabel("Seconds")

    x = range(len(scenarios))
    axes[1, 0].bar(x, correct, color="#7bb662", label="Correct")
    axes[1, 0].bar(x, wrong, bottom=correct, color="#d96557", label="Wrong")
    axes[1, 0].set_xticks(list(x), scenarios)
    axes[1, 0].set_title("Correct vs Wrong")
    axes[1, 0].set_ylabel("Questions")
    axes[1, 0].legend()

    if root_causes:
        labels = list(root_causes.keys())
        values = list(root_causes.values())
        axes[1, 1].barh(labels, values, color="#9c755f")
        axes[1, 1].set_title("Failure Reasons Across Scenarios")
        axes[1, 1].set_xlabel("Count")
    else:
        axes[1, 1].text(0.5, 0.5, "No failures!", ha="center", va="center", fontsize=14)
        axes[1, 1].set_title("Failure Reasons Across Scenarios")
        axes[1, 1].set_axis_off()

    fig.text(
        0.5,
        0.01,
        (
            f"Total questions: {total} | Correct: {total_correct} | Wrong: {total_wrong} | "
            f"Scenarios: simple_lookup, aggregation, filtering_ranking"
        ),
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows, root_causes = _collect_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combined = {
        "scenarios": rows,
        "overall": {
            "total": sum(r["total"] for r in rows),
            "correct": sum(r["correct"] for r in rows),
            "wrong": sum(r["wrong"] for r in rows),
            "skipped": sum(r["skipped"] for r in rows),
            "accuracy_pct": round(
                100 * sum(r["correct"] for r in rows)
                / max(sum(r["correct"] + r["wrong"] for r in rows), 1),
                1,
            ),
        },
        "root_causes": dict(root_causes),
    }

    (OUT_DIR / "combined_eval_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _build_dashboard(rows, root_causes, OUT_DIR / "combined_eval_dashboard.png")

    print(json.dumps(combined["overall"], indent=2))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
