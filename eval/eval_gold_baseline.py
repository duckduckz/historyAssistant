"""
Evaluate the full chatbot pipeline against gold_baseline_filled.csv.

Usage (from repo root):
    python eval/eval_gold_baseline.py

Outputs (under 07_outputs/eval_gold_baseline/):
    - results.json          per-question detail
    - summary.json          aggregate metrics
    - dashboard.png         accuracy visualisation
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from elasticsearch import ElasticsearchWarning

warnings.filterwarnings("ignore", category=ElasticsearchWarning)

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

BASELINE_CSV = _ROOT / "eval/gold_baseline_filled.csv"
OUT_DIR = _ROOT / "eval/outputs/eval_gold_baseline"
RESULTS_JSON = OUT_DIR / "results.json"
SUMMARY_JSON = OUT_DIR / "summary.json"
DASHBOARD_PNG = OUT_DIR / "dashboard.png"
BY_CATEGORY_DIR = OUT_DIR / "by_category"

TOLERANCE_BY_METRIC: Dict[str, float] = {
    "avg_response_time": 0.5,
    "median_response_time": 1.0,
    "avg_staying_time": 0.1,
    "response_within_2_min_rate": 0.5,
    "response_within_3_min_rate": 0.5,
    "response_over_5_min_rate": 0.5,
}
DEFAULT_TOLERANCE = 0.5

SCENARIO_ORDER = ["SL", "AG", "RF", "CP", "TR", "CH"]
SCENARIO_LABELS = {
    "SL": "Simple Lookup",
    "AG": "Aggregation",
    "RF": "Ranking/Filter",
    "CP": "Comparison",
    "TR": "Trend",
    "CH": "Chart",
}


# ── value helpers ────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _safe_str(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    return str(v).strip()


def _num_close(a: Any, b: Any, tol: float) -> bool:
    fa, fb = _safe_float(a), _safe_float(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) <= tol


def _parse_kv_list(text: str) -> List[Tuple[str, str]]:
    """Parse 'K1:V1; K2:V2' into [(K1,V1),(K2,V2)]."""
    pairs = []
    for part in re.split(r";\s*", text.strip()):
        part = part.strip()
        if ":" in part:
            k, _, v = part.partition(":")
            pairs.append((k.strip(), v.strip()))
    return pairs


def _parse_comparison_ref(text: str) -> Dict[str, Any]:
    """Parse 'side_a=X; side_b=Y; winner=Z' into a dict."""
    result: Dict[str, Any] = {}
    for part in re.split(r";\s*", text.strip()):
        if "=" in part:
            k, _, v = part.partition("=")
            result[k.strip()] = v.strip()
    return result


# ── per-scenario comparison ───────────────────────────────────────────────────

def _tolerance_for_row(row: pd.Series) -> float:
    metric = _safe_str(row.get("expected_metric", ""))
    for k, t in TOLERANCE_BY_METRIC.items():
        if k in metric:
            return t
    return DEFAULT_TOLERANCE


def compare_simple_lookup(exec_result: Dict, row: pd.Series) -> Tuple[Optional[bool], str]:
    expected_n = _safe_float(row.get("expected_numeric_value"))
    if expected_n is None:
        return None, "no expected_numeric_value"
    if not exec_result.get("ok"):
        return False, f"execution error: {exec_result.get('error','')}"
    actual = exec_result.get("total_count")
    if actual is None:
        actual = len(exec_result.get("records") or [])
    ok = _num_close(actual, expected_n, 0)
    return ok, f"got={actual} want={int(expected_n)}"


def compare_aggregation(exec_result: Dict, row: pd.Series) -> Tuple[Optional[bool], str]:
    if not exec_result.get("ok"):
        return False, f"execution error: {exec_result.get('error','')}"
    tol = _tolerance_for_row(row)

    # grouped / ranking — check expected_top_items
    top_items_str = _safe_str(row.get("expected_top_items"))
    if top_items_str:
        records = exec_result.get("records") or []
        expected_pairs = _parse_kv_list(top_items_str)
        if not expected_pairs:
            return None, "cannot parse expected_top_items"
        # build a lookup from actual records: first non-metric column → metric_value
        actual_map: Dict[str, float] = {}
        for r in records:
            keys = [k for k in r if k != "metric_value"]
            if keys:
                key_val = str(r[keys[0]])
                actual_map[key_val.lower()] = _safe_float(r.get("metric_value")) or 0.0
        mismatches = []
        for k, v in expected_pairs:
            exp_v = _safe_float(v)
            act_v = actual_map.get(k.lower())
            if act_v is None or exp_v is None or not _num_close(act_v, exp_v, tol):
                mismatches.append(f"{k}: got={act_v} want={exp_v}")
        if mismatches:
            return False, "; ".join(mismatches[:3])
        return True, f"grouped match ({len(expected_pairs)} groups)"

    # scalar
    expected_n = _safe_float(row.get("expected_numeric_value"))
    if expected_n is None:
        return None, "no expected value"
    scalar = exec_result.get("scalar_answer")
    if scalar is None:
        records = exec_result.get("records") or []
        if records:
            scalar = records[0].get("metric_value")
    ok = _num_close(scalar, expected_n, tol)
    return ok, f"got={scalar} want={expected_n}"


def compare_ranking(exec_result: Dict, row: pd.Series) -> Tuple[Optional[bool], str]:
    return compare_aggregation(exec_result, row)


def compare_comparison(exec_result: Dict, row: pd.Series) -> Tuple[Optional[bool], str]:
    if not exec_result.get("ok"):
        return False, f"execution error: {exec_result.get('error','')}"
    ref_str = _safe_str(row.get("expected_answer_reference"))
    if not ref_str:
        return None, "no expected_answer_reference"
    expected = _parse_comparison_ref(ref_str)
    tol = _tolerance_for_row(row)

    # check main value (side_a)
    comp_summary = exec_result.get("comparison_summary") or {}
    main_val = comp_summary.get("main_value")
    comp_val = comp_summary.get("comparison_value")
    winner = comp_summary.get("winner_label") or exec_result.get("scalar_answer")

    errors = []

    exp_a = _safe_float(expected.get("side_a"))
    if exp_a is not None and main_val is not None:
        if not _num_close(main_val, exp_a, tol):
            errors.append(f"side_a: got={main_val} want={exp_a}")

    exp_b = _safe_float(expected.get("side_b"))
    if exp_b is not None and comp_val is not None:
        if not _num_close(comp_val, exp_b, tol):
            errors.append(f"side_b: got={comp_val} want={exp_b}")

    # winner: baseline uses "side_a"/"side_b"; system returns a label string
    exp_winner = expected.get("winner")
    if exp_winner and winner:
        # map "side_a" → main_label, "side_b" → comp_label
        main_label = str(comp_summary.get("main_label") or "side_a").lower()
        comp_label = str(comp_summary.get("comparison_label") or "side_b").lower()
        winner_str = str(winner).lower()
        expected_is_main = exp_winner in ("side_a",)
        actual_is_main = winner_str in (main_label, "side_a") or (
            "side_b" not in winner_str and "comp" not in winner_str
        )
        # simpler: compare which side has the higher value
        if exp_a is not None and exp_b is not None:
            exp_main_wins = (exp_winner == "side_a")
            if main_val is not None and comp_val is not None:
                act_main_wins = float(main_val) >= float(comp_val)
                if exp_main_wins != act_main_wins:
                    errors.append(f"winner: got={'side_a' if act_main_wins else 'side_b'} want={exp_winner}")

    if errors:
        return False, "; ".join(errors)
    if main_val is None and comp_val is None:
        return None, "no comparison values in result"
    return True, f"main={main_val} comp={comp_val}"


def compare_trend(exec_result: Dict, row: pd.Series) -> Tuple[Optional[bool], str]:
    if not exec_result.get("ok"):
        return False, f"execution error: {exec_result.get('error','')}"
    ref_str = _safe_str(row.get("expected_answer_reference"))
    # expected_answer_reference = "month trend over N buckets"
    bucket_match = re.search(r"over\s+(\d+)\s+bucket", ref_str)
    expected_buckets = int(bucket_match.group(1)) if bucket_match else None

    records = exec_result.get("records") or []
    actual_buckets = len(records)

    top_items_str = _safe_str(row.get("expected_top_items"))
    tol = _tolerance_for_row(row)

    errors = []
    if expected_buckets is not None and actual_buckets != expected_buckets:
        errors.append(f"bucket count: got={actual_buckets} want={expected_buckets}")

    if top_items_str and records:
        sample_pairs = _parse_kv_list(top_items_str)[:5]
        # build a lookup by the first non-metric column value
        actual_map: Dict[str, Any] = {}
        for r in records:
            keys = [k for k in r if k != "metric_value"]
            if keys:
                actual_map[str(r[keys[0]])] = _safe_float(r.get("metric_value"))
        for k, v in sample_pairs:
            exp_v = _safe_float(v)
            act_v = actual_map.get(k)
            if act_v is None or exp_v is None or not _num_close(act_v, exp_v, tol):
                errors.append(f"bucket {k}: got={act_v} want={exp_v}")

    if errors:
        return False, "; ".join(errors[:3])
    detail = f"{actual_buckets} buckets"
    if expected_buckets:
        detail += f" (expected {expected_buckets})"
    return True, detail


def compare_chart(exec_result: Dict, row: pd.Series) -> Tuple[Optional[bool], str]:
    if not exec_result.get("ok"):
        return False, f"execution error: {exec_result.get('error','')}"
    chart_payload = exec_result.get("chart_payload")
    if not chart_payload:
        return False, "no chart_payload produced"

    expected_type = _safe_str(row.get("expected_chart_type")).lower()
    if expected_type and expected_type not in ("", "nan"):
        actual_type = str(
            (chart_payload.get("chart_spec") or {}).get("chart_type") or ""
        ).strip().lower()
        if actual_type != expected_type:
            return False, f"chart_type: got={actual_type} want={expected_type}"

    # spot-check data values if provided
    top_items_str = _safe_str(row.get("expected_top_items"))
    if top_items_str:
        records = chart_payload.get("rows") or []
        tol = _tolerance_for_row(row)
        expected_pairs = _parse_kv_list(top_items_str)[:5]
        actual_map: Dict[str, Any] = {}
        for r in records:
            keys = [k for k in r if k != "metric_value"]
            if keys:
                actual_map[str(r[keys[0]]).lower()] = _safe_float(r.get("metric_value"))
        mismatches = []
        for k, v in expected_pairs:
            exp_v = _safe_float(v)
            act_v = actual_map.get(k.lower())
            if act_v is None or exp_v is None or not _num_close(act_v, exp_v, tol):
                mismatches.append(f"{k}: got={act_v} want={exp_v}")
        if mismatches:
            return False, "data mismatch: " + "; ".join(mismatches[:2])

    return True, f"chart_type={expected_type or 'any'}"


def compare_result(exec_result: Dict, row: pd.Series) -> Tuple[Optional[bool], str]:
    prefix = _safe_str(row.get("eval_id", ""))[:2].upper()
    if prefix == "SL":
        return compare_simple_lookup(exec_result, row)
    if prefix == "AG":
        return compare_aggregation(exec_result, row)
    if prefix == "RF":
        return compare_ranking(exec_result, row)
    if prefix == "CP":
        return compare_comparison(exec_result, row)
    if prefix == "TR":
        return compare_trend(exec_result, row)
    if prefix == "CH":
        return compare_chart(exec_result, row)
    return None, f"unknown scenario prefix: {prefix}"


def _scenario_prefix(qid: str) -> str:
    return _safe_str(qid)[:2].upper()


def _tag_root_cause(exec_result: Dict, match_ok: Optional[bool]) -> Optional[str]:
    if match_ok is True:
        return None
    if exec_result.get("guardrail_blocked"):
        return "guardrail_blocked"
    scenario_pred = exec_result.get("scenario_predicted")
    scenario_exp = exec_result.get("scenario_expected")
    if scenario_pred and scenario_exp and scenario_pred != scenario_exp:
        return "scenario_mismatch"
    if not exec_result.get("ok"):
        return "execution_failed"
    return "answer_mismatch"


# ── pipeline runner ───────────────────────────────────────────────────────────

def run_question(question: str, handle_chat_turn: Any) -> Tuple[Dict, float]:
    t0 = time.monotonic()
    try:
        _, _, debug_payload, _, _, _ = handle_chat_turn(
            question, [], "", "", "", "", "none", True,
        )
    except Exception as exc:
        elapsed = round(time.monotonic() - t0, 3)
        return {"ok": False, "error": str(exc), "debug_payload": None}, elapsed

    elapsed = round(time.monotonic() - t0, 3)
    if not debug_payload:
        return {"ok": False, "error": "no debug_payload", "debug_payload": None}, elapsed

    execution_items = debug_payload.get("execution") or []
    if not execution_items:
        return {"ok": False, "error": "no execution items", "debug_payload": debug_payload}, elapsed

    exec_result = execution_items[0].get("execution_result") or {}
    exec_result["debug_payload"] = debug_payload

    planning_items = debug_payload.get("planning") or []
    if planning_items:
        exec_result["scenario_predicted"] = planning_items[0].get("scenario")

    return exec_result, elapsed


# ── dashboard ─────────────────────────────────────────────────────────────────

_PASS_COLOR  = "#4ade80"   # green
_FAIL_COLOR  = "#f87171"   # red
_SKIP_COLOR  = "#d1d5db"   # gray
_BG          = "#f8fafc"
_PANEL_BG    = "#ffffff"
_ACCENT      = "#6366f1"   # indigo


def _accuracy_pct(rows: List[Dict]) -> float:
    c = sum(1 for r in rows if r.get("match") is True)
    w = sum(1 for r in rows if r.get("match") is False)
    return 100 * c / max(c + w, 1)


def build_dashboard(results: List[Dict], out_path: Path) -> None:
    total   = len(results)
    correct = sum(1 for r in results if r.get("match") is True)
    wrong   = sum(1 for r in results if r.get("match") is False)
    skipped = sum(1 for r in results if r.get("match") is None)
    overall_acc = 100 * correct / max(correct + wrong, 1)

    fig = plt.figure(figsize=(18, 13), facecolor=_BG)
    fig.suptitle(
        f"Gold Baseline Accuracy Report   —   {total} questions   |   {overall_acc:.1f}% overall accuracy",
        fontsize=16, fontweight="bold", color="#1e293b", y=0.97,
    )

    gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.38,
                          left=0.06, right=0.97, top=0.91, bottom=0.06)

    ax_donut   = fig.add_subplot(gs[0, 0])
    ax_by_cat  = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[0, 2])
    ax_latency = fig.add_subplot(gs[1, :2])
    ax_cause   = fig.add_subplot(gs[1, 2])
    ax_detail  = fig.add_subplot(gs[2, :])

    for ax in fig.get_axes():
        ax.set_facecolor(_PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#e2e8f0")

    # ── 1. Donut ──────────────────────────────────────────────────────────
    wedge_sizes = [correct, wrong, skipped]
    wedge_colors = [_PASS_COLOR, _FAIL_COLOR, _SKIP_COLOR]
    wedge_labels = [f"Pass ({correct})", f"Fail ({wrong})", f"Skip ({skipped})"]
    wedges, _ = ax_donut.pie(
        wedge_sizes, colors=wedge_colors, startangle=90,
        wedgeprops={"width": 0.52, "edgecolor": _BG, "linewidth": 2},
    )
    ax_donut.text(0, 0, f"{overall_acc:.0f}%", ha="center", va="center",
                  fontsize=22, fontweight="bold", color="#1e293b")
    ax_donut.legend(wedges, wedge_labels, loc="lower center",
                    bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8,
                    frameon=False)
    ax_donut.set_title("Overall Pass / Fail", fontsize=11, fontweight="bold",
                        color="#334155", pad=8)

    # ── 2. Accuracy by category ───────────────────────────────────────────
    cat_names, cat_accs, cat_correct, cat_total = [], [], [], []
    for prefix in SCENARIO_ORDER:
        subset = [r for r in results if str(r.get("eval_id", "")).startswith(prefix)]
        if not subset:
            continue
        acc = _accuracy_pct(subset)
        cat_names.append(SCENARIO_LABELS[prefix])
        cat_accs.append(acc)
        cat_correct.append(sum(1 for r in subset if r.get("match") is True))
        cat_total.append(len(subset))

    bar_colors = [_PASS_COLOR if a >= 80 else (_FAIL_COLOR if a < 50 else "#fbbf24")
                  for a in cat_accs]
    bars = ax_by_cat.bar(cat_names, cat_accs, color=bar_colors, edgecolor=_BG,
                          linewidth=1.5, width=0.6)
    for bar, c, tot, acc in zip(bars, cat_correct, cat_total, cat_accs):
        ax_by_cat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{c}/{tot}", ha="center", va="bottom", fontsize=8, color="#334155")
    ax_by_cat.axhline(100, color="#e2e8f0", linestyle="--", linewidth=0.8)
    ax_by_cat.set_ylim(0, 115)
    ax_by_cat.set_ylabel("Accuracy (%)", fontsize=9)
    ax_by_cat.tick_params(axis="x", labelsize=8, rotation=15)
    ax_by_cat.tick_params(axis="y", labelsize=8)
    ax_by_cat.set_title("Accuracy by Scenario Category", fontsize=11,
                          fontweight="bold", color="#334155", pad=8)

    # ── 3. Pass/Fail heatmap (per question ID) ────────────────────────────
    ids = [r["eval_id"] for r in results]
    match_vals = [1 if r.get("match") is True else (0 if r.get("match") is False else 0.5)
                  for r in results]
    n = len(ids)
    cols = 10
    rows_grid = (n + cols - 1) // cols
    grid = np.full((rows_grid, cols), np.nan)
    for i, v in enumerate(match_vals):
        grid[i // cols, i % cols] = v
    cmap = matplotlib.colors.ListedColormap([_FAIL_COLOR, _SKIP_COLOR, _PASS_COLOR])
    ax_heatmap.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    # label each cell
    for i, (qid, v) in enumerate(zip(ids, match_vals)):
        r_idx, c_idx = i // cols, i % cols
        short = qid[:4]
        color = "#1e293b" if v == 0.5 else ("white" if v == 1 else "white")
        ax_heatmap.text(c_idx, r_idx, short, ha="center", va="center",
                         fontsize=6.5, color=color, fontweight="bold")
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_heatmap.set_title("Per-Question Pass/Fail Grid", fontsize=11,
                           fontweight="bold", color="#334155", pad=8)
    legend_patches = [
        mpatches.Patch(color=_PASS_COLOR, label="Pass"),
        mpatches.Patch(color=_FAIL_COLOR, label="Fail"),
        mpatches.Patch(color=_SKIP_COLOR, label="Skip"),
    ]
    ax_heatmap.legend(handles=legend_patches, loc="lower right",
                       bbox_to_anchor=(1.0, -0.18), ncol=3, fontsize=7, frameon=False)

    # ── 4. Latency per question ───────────────────────────────────────────
    latencies = [r.get("elapsed_sec", 0) for r in results]
    lat_colors = [_PASS_COLOR if r.get("match") is True else
                  (_FAIL_COLOR if r.get("match") is False else _SKIP_COLOR)
                  for r in results]
    x_pos = np.arange(len(ids))
    ax_latency.bar(x_pos, latencies, color=lat_colors, width=0.75, edgecolor=_BG, linewidth=0.5)
    ax_latency.set_xticks(x_pos)
    ax_latency.set_xticklabels(ids, rotation=70, ha="right", fontsize=7)
    ax_latency.set_ylabel("Seconds", fontsize=9)
    avg_lat = np.mean(latencies) if latencies else 0
    ax_latency.axhline(avg_lat, color=_ACCENT, linestyle="--", linewidth=1.2,
                        label=f"Avg {avg_lat:.1f}s")
    ax_latency.legend(fontsize=8, frameon=False)
    ax_latency.tick_params(axis="y", labelsize=8)
    ax_latency.set_title("Response Latency per Question (green=pass, red=fail)",
                           fontsize=11, fontweight="bold", color="#334155", pad=8)

    # ── 5. Failure breakdown ──────────────────────────────────────────────
    from collections import Counter
    root_causes = Counter(r.get("root_cause") for r in results if r.get("root_cause"))
    if root_causes:
        labels_rc = list(root_causes.keys())
        vals_rc   = list(root_causes.values())
        rc_colors = ["#f87171", "#fb923c", "#fbbf24", "#a78bfa", "#60a5fa"][:len(labels_rc)]
        ax_cause.barh(labels_rc, vals_rc, color=rc_colors, edgecolor=_BG, linewidth=1)
        for i, v in enumerate(vals_rc):
            ax_cause.text(v + 0.1, i, str(v), va="center", fontsize=9)
        ax_cause.set_xlim(0, max(vals_rc) * 1.3)
        ax_cause.tick_params(axis="y", labelsize=9)
        ax_cause.tick_params(axis="x", labelsize=8)
    else:
        ax_cause.text(0.5, 0.5, "🎉  No failures!", ha="center", va="center",
                       fontsize=13, color="#4ade80")
        ax_cause.set_axis_off()
    ax_cause.set_title("Failure Root Causes", fontsize=11, fontweight="bold",
                         color="#334155", pad=8)

    # ── 6. Detail table (bottom strip) ────────────────────────────────────
    ax_detail.set_axis_off()
    col_headers = ["ID", "Scenario", "Question (truncated)", "Match", "Detail", "Sec"]
    table_data = [
        [
            r["eval_id"],
            r.get("scenario_subtype", "")[:12],
            str(r.get("question", ""))[:55],
            {True: "✓", False: "✗", None: "—"}[r.get("match")],
            str(r.get("match_detail", ""))[:45],
            f"{r.get('elapsed_sec', 0):.1f}",
        ]
        for r in results
    ]
    row_colors = [
        [(_PASS_COLOR if r.get("match") is True else
          (_FAIL_COLOR if r.get("match") is False else _SKIP_COLOR))] + ["#f8fafc"] * 5
        for r in results
    ]
    tbl = ax_detail.table(
        cellText=table_data,
        colLabels=col_headers,
        cellLoc="left",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.8)
    tbl.scale(1, 1.25)
    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row_i == 0:
            cell.set_facecolor(_ACCENT)
            cell.set_text_props(color="white", fontweight="bold")
    ax_detail.set_title("Per-Question Results", fontsize=11, fontweight="bold",
                          color="#334155", pad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  Dashboard saved → {out_path}")


# main

def _json_default(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, pd.DataFrame):
        return f"<DataFrame {v.shape}>"
    return str(v)


def _result_sort_key(result: Dict[str, Any]) -> Tuple[int, Any]:
    qid = _safe_str(result.get("eval_id"))
    prefix = _scenario_prefix(qid)
    try:
        prefix_idx = SCENARIO_ORDER.index(prefix)
    except ValueError:
        prefix_idx = len(SCENARIO_ORDER)
    match = re.search(r"(\d+)$", qid)
    suffix_num = int(match.group(1)) if match else 0
    return (prefix_idx, suffix_num, qid)


def _load_existing_results() -> List[Dict[str, Any]]:
    if not RESULTS_JSON.exists():
        return []
    try:
        data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not load existing results.json: {exc}")
        return []
    if not isinstance(data, list):
        print("Existing results.json is not a list; ignoring it.")
        return []
    cleaned = [item for item in data if isinstance(item, dict) and _safe_str(item.get("eval_id"))]
    return sorted(cleaned, key=_result_sort_key)


def _should_use_cached_result(result: Dict[str, Any], rerun_failures: bool) -> bool:
    if not result:
        return False
    if result.get("root_cause") == "execution_failed":
        return False
    if rerun_failures and result.get("match") is not True:
        return False
    return True


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = sum(1 for r in results if r["match"] is True)
    wrong = sum(1 for r in results if r["match"] is False)
    skipped = sum(1 for r in results if r["match"] is None)
    avg_lat = sum(r["elapsed_sec"] for r in results) / max(len(results), 1)

    category_summary = {}
    for prefix in SCENARIO_ORDER:
        subset = [r for r in results if _scenario_prefix(r.get("eval_id")) == prefix]
        if subset:
            c = sum(1 for r in subset if r["match"] is True)
            w = sum(1 for r in subset if r["match"] is False)
            s = sum(1 for r in subset if r["match"] is None)
            category_summary[SCENARIO_LABELS[prefix]] = {
                "prefix": prefix,
                "total": len(subset),
                "correct": c,
                "wrong": w,
                "skipped": s,
                "accuracy_pct": round(100 * c / max(c + w, 1), 1),
                "average_latency_sec": round(
                    sum(r["elapsed_sec"] for r in subset) / max(len(subset), 1), 2
                ),
            }

    return {
        "total": len(results),
        "correct": correct,
        "wrong": wrong,
        "skipped": skipped,
        "accuracy_pct": round(100 * correct / max(correct + wrong, 1), 1),
        "average_latency_sec": round(avg_lat, 2),
        "by_category": category_summary,
    }


def _write_outputs(results: List[Dict[str, Any]]) -> None:
    ordered_results = sorted(results, key=_result_sort_key)
    summary = _build_summary(ordered_results)

    RESULTS_JSON.write_text(
        json.dumps(ordered_results, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    BY_CATEGORY_DIR.mkdir(parents=True, exist_ok=True)
    for prefix in SCENARIO_ORDER:
        subset = [r for r in ordered_results if _scenario_prefix(r.get("eval_id")) == prefix]
        if not subset:
            continue
        payload = {
            "prefix": prefix,
            "label": SCENARIO_LABELS[prefix],
            "summary": summary["by_category"].get(SCENARIO_LABELS[prefix], {}),
            "results": subset,
        }
        (BY_CATEGORY_DIR / f"{prefix.lower()}_results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    build_dashboard(ordered_results, DASHBOARD_PNG)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the chatbot against the gold baseline.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore saved results and rerun every question from scratch.",
    )
    parser.add_argument(
        "--rerun-failures",
        action="store_true",
        help="Reuse cached passes only; rerun cached fails and skips.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from historyAssistant.app.chatbot_service import handle_chat_turn

    baseline = pd.read_csv(BASELINE_CSV, encoding="utf-8-sig")
    print(f"Loaded {len(baseline)} questions from {BASELINE_CSV.name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = [] if args.fresh else _load_existing_results()
    results_by_qid: Dict[str, Dict[str, Any]] = {
        _safe_str(r.get("eval_id")): r for r in results
    }
    if args.fresh:
        print("Fresh run requested; ignoring saved cache.")
    elif results_by_qid:
        print(f"Resuming from {len(results_by_qid)} saved result(s) in {RESULTS_JSON.name}")

    for idx, row in baseline.iterrows():
        qid = _safe_str(row.get("eval_id", f"Q{idx}"))
        question = _safe_str(row.get("question_or_conversation", ""))
        scenario = _safe_str(row.get("scenario", ""))

        cached = results_by_qid.get(qid)
        if _should_use_cached_result(cached, rerun_failures=args.rerun_failures):
            cached = results_by_qid[qid]
            cached_status = {True: "PASS", False: "FAIL", None: "SKIP"}[cached.get("match")]
            print(f"[{idx+1:02d}/{len(baseline)}] {qid:6s} | cached {cached_status:4s} | {question[:70]}")
            continue

        print(f"[{idx+1:02d}/{len(baseline)}] {qid:6s} | {question[:70]}")

        exec_result, elapsed = run_question(question, handle_chat_turn)
        match_ok, match_detail = compare_result(exec_result, row)
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[match_ok]
        print(f"         {status:4s} | {elapsed:.1f}s | {match_detail}")

        result = {
            "eval_id":         qid,
            "scenario":        scenario,
            "scenario_subtype": _safe_str(row.get("subtype", "")),
            "question":        question,
            "match":           match_ok,
            "match_detail":    match_detail,
            "elapsed_sec":     elapsed,
            "root_cause":      _tag_root_cause(exec_result, match_ok),
            "scenario_predicted": exec_result.get("scenario_predicted"),
        }
        if qid in results_by_qid:
            results = [r for r in results if _safe_str(r.get("eval_id")) != qid]
        results.append(result)
        results_by_qid[qid] = result
        _write_outputs(results)

    summary = _build_summary(results)
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
