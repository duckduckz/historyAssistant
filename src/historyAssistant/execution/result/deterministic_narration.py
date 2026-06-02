from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from historyAssistant.execution.result.display_labels import FIELD_LABELS
from historyAssistant.execution.result.raw_record_answers import (
    detect_lookup_projection,
    detect_requested_record_field,
    format_raw_record_list,
)
from historyAssistant.execution.result.narration_field_selector import NarrationFieldSelection, display_fields_from_selection

YamlDict = Dict[str, Any]


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def display_label(field_name: str) -> str:
    return FIELD_LABELS.get(field_name, field_name.replace("_", " ").strip().title())


def format_execution_error(error: Any) -> str:
    error_text = str(error or "").strip()
    if "Failed to establish a new connection" in error_text or "WinError 10061" in error_text:
        return (
            "I could build the query, but Elasticsearch is not reachable at `localhost:9200`. "
            "Please start Elasticsearch or check the host/port in `configs/shared/app_config.yaml`, then try again."
        )
    return f"Execution failed: {error_text}"


def deterministic_narration(
    question: str,
    execution_result: YamlDict,
    *,
    field_selection: NarrationFieldSelection | None = None,
) -> str:
    if execution_result.get("error"):
        return format_execution_error(execution_result["error"])

    ranked_single = try_template_ranked_single_result(execution_result)
    if ranked_single:
        return ranked_single

    scalar_answer = execution_result.get("scalar_answer")
    records = execution_result.get("records", []) or []
    total_count = int(execution_result.get("total_count", len(records) or 0))
    result_type = execution_result.get("result_type", "aggregated")
    comparison = execution_result.get("comparison")
    warnings = list(execution_result.get("warnings", []) or [])
    comparison_change = comparison.get("change", {}) if isinstance(comparison, dict) else {}
    grouped_changes = comparison_change.get("grouped_changes") if isinstance(comparison_change, dict) else None

    if comparison and comparison.get("ok") and grouped_changes:
        preview_count = min(10, len(grouped_changes))
        preview_rows = grouped_changes[:preview_count]
        plan = execution_result.get("plan") or {}
        group_by = list(plan.get("group_by") or [])
        group_key = group_by[0] if group_by else None
        if not group_key and preview_rows:
            group_key = next(
                (
                    key
                    for key in preview_rows[0]
                    if key not in {"metric_value_main", "metric_value_comp", "absolute_change", "percent_change"}
                ),
                None,
            )

        lines = [f"Comparison by {display_label(group_key).lower()}:" if group_key else "Comparison results:", ""]
        for idx, row in enumerate(preview_rows, start=1):
            group_value = row.get(group_key) if group_key else None
            main_value = row.get("metric_value_main")
            comp_value = row.get("metric_value_comp")
            absolute_change = row.get("absolute_change")
            percent_change = row.get("percent_change")
            change_text = format_value(absolute_change)
            if percent_change is not None and pd.notna(percent_change):
                change_text += f" ({format_value(percent_change)}%)"
            if group_value not in (None, ""):
                lines.append(
                    f"{idx}. {format_value(group_value)} - main {format_value(main_value)}, "
                    f"comparison {format_value(comp_value)}, change {change_text}"
                )
        if len(grouped_changes) > preview_count:
            lines.append("")
            lines.append(f"Showing {preview_count} of {len(grouped_changes)} comparable groups.")
        if comparison_change.get("partial_overlap"):
            lines.append("Some groups existed in only one comparison period, so they are not included in the change rows.")
        if warnings:
            lines.append("\nNote: " + " ".join(warnings))
        return "\n".join(lines).rstrip()

    if scalar_answer is not None:
        answer = f"Result: {format_value(scalar_answer)}"
        if comparison and comparison.get("ok"):
            comp_scalar = comparison.get("scalar_answer")
            if comp_scalar is not None:
                answer += f"\nComparison: {format_value(comp_scalar)}"
            change = comparison.get("change", {})
            if change.get("has_data"):
                delta = change.get("absolute_change")
                pct = change.get("percent_change")
                direction = change.get("direction")
                detail = f"Change: {format_value(delta)}"
                if pct is not None:
                    detail += f" ({format_value(pct)}%)"
                if direction:
                    detail += f", {direction}"
                answer += f"\n{detail}"
        if warnings:
            answer += "\n\nNote: " + " ".join(warnings)
        return answer

    if not records:
        if execution_result.get("empty"):
            answer = "No matching results found."
            if warnings:
                answer += "\n\nNote: " + " ".join(warnings)
            return answer
        return "No grounded answer was available."

    preview_count = min(10, len(records))
    preview_rows = records[:preview_count]

    if result_type == "raw_records":
        if total_count == 1 and preview_rows:
            if field_selection and field_selection.used and field_selection.requested_field:
                requested_field = field_selection.requested_field
                requested_label = field_selection.requested_label or display_label(requested_field)
            else:
                requested_field, requested_label = detect_requested_record_field(question, preview_rows[0])
            if requested_field and preview_rows[0].get(requested_field) not in (None, "", "unknown"):
                answer_lines = [
                    f"{requested_label}: {format_value(preview_rows[0][requested_field])}",
                    "",
                    "Matching call details:",
                    "",
                    format_raw_record_list(
                        preview_rows,
                        total_count,
                        preview_count,
                        display_fields=display_fields_from_selection(field_selection),
                    ),
                ]
                return "\n".join(answer_lines).rstrip()
        return format_raw_record_list(
            preview_rows,
            total_count,
            preview_count,
            display_fields=display_fields_from_selection(field_selection),
        )

    projection_field, projection_label = detect_lookup_projection(question, preview_rows or records)
    if projection_field and len(records) > 0 and projection_field in records[0]:
        projection_label = display_label(projection_field)
        has_metric = any("metric_value" in row for row in records)
        if has_metric:
            lines = [f"Found {len(records)} matching {projection_label.lower()} value(s), with matching record counts.", ""]
        else:
            lines = [f"Found {len(records)} matching {projection_label.lower()} value(s).", ""]
        for idx, row in enumerate(records[:10], start=1):
            value = row.get(projection_field)
            if value not in (None, "", "unknown"):
                metric_value = row.get("metric_value")
                suffix = f" ({format_value(metric_value)} record(s))" if has_metric and metric_value not in (None, "") else ""
                lines.append(f"{idx}. {format_value(value)}{suffix}")
        if len(records) > 10:
            lines.append("")
            lines.append(f"Showing 10 of {len(records)} {projection_label.lower()} values.")
        return "\n".join(lines).rstrip()

    plan = execution_result.get("plan") or {}
    ranking_order = plan.get("ranking_order") or plan.get("order_by", {})
    if isinstance(ranking_order, dict):
        ranking_direction = str(ranking_order.get("direction", "desc")).lower()
    else:
        ranking_direction = "desc"
    is_ranked = bool(plan.get("ranking_order") or plan.get("order_by"))

    if preview_rows:
        row_keys = list(preview_rows[0].keys())
        metric_key = None
        dimension_key = None
        for k in reversed(row_keys):
            v = preview_rows[0].get(k)
            if isinstance(v, (int, float)) and v not in (None, ""):
                metric_key = k
                break
        for k in row_keys:
            if k != metric_key:
                dimension_key = k
                break

        if dimension_key and metric_key:
            metric_label = display_label(metric_key)
            dimension_label = display_label(dimension_key)
            direction_word = "least" if ranking_direction == "asc" else "most"
            intro = (
                f"Here are the {direction_word} results by {dimension_label.lower()} ({metric_label}):"
                if is_ranked
                else f"Results by {dimension_label.lower()} ({metric_label}):"
            )
            lines = [intro, ""]
            for idx, row in enumerate(preview_rows, start=1):
                dim_val = row.get(dimension_key)
                metric_val = row.get(metric_key)
                if dim_val not in (None, "") and metric_val not in (None, ""):
                    lines.append(f"{idx}. {format_value(dim_val)} - {format_value(metric_val)}")
            if len(records) > preview_count:
                lines.append(f"\nShowing {preview_count} of {len(records)} results.")
            if comparison and comparison.get("ok") and comparison.get("change", {}).get("suppressed"):
                lines.append(f"Comparison note: {comparison['change'].get('reason')}")
            if warnings:
                lines.append("\nNote: " + " ".join(warnings))
            return "\n".join(lines)

    lines = [f"Returned {len(records)} row(s)."]
    for row in preview_rows:
        parts = [f"{display_label(key)}: {format_value(value)}" for key, value in row.items() if value not in (None, "")]
        if parts:
            lines.append("- " + ", ".join(parts))
    if comparison and comparison.get("ok") and comparison.get("change", {}).get("suppressed"):
        lines.append(f"Comparison note: {comparison['change'].get('reason')}")
    if len(records) > preview_count:
        lines.append(f"Showing {preview_count} of {len(records)} rows.")
    return "\n".join(lines)


def try_template_ranked_single_result(execution_result: YamlDict) -> str | None:
    if execution_result.get("error") or execution_result.get("warnings"):
        return None
    if execution_result.get("result_type", "aggregated") != "aggregated":
        return None
    if execution_result.get("comparison"):
        return None

    plan = execution_result.get("plan") or {}
    if plan.get("bucket_interval"):
        return None
    group_by = list(plan.get("group_by") or [])
    if len(group_by) != 1:
        return None

    ranking_order = plan.get("ranking_order") if isinstance(plan.get("ranking_order"), dict) else None
    if not ranking_order or ranking_order.get("field") != "metric_value":
        return None
    target_group = ranking_order.get("target_group") or group_by[0]
    if target_group != group_by[0]:
        return None

    records = execution_result.get("records") or []
    if len(records) != 1:
        return None
    row = records[0]
    if target_group not in row or "metric_value" not in row:
        return None

    group_value = row.get(target_group)
    metric_value = row.get("metric_value")
    if group_value in (None, "") or metric_value in (None, ""):
        return None

    direction = str(ranking_order.get("direction") or "desc").lower()
    result_word = "Lowest" if direction == "asc" else "Top"
    group_label = display_label(target_group).lower()
    metric_label = display_label(str(plan.get("metric") or "metric_value")).lower()
    return (
        f"{result_word} result by {group_label}: "
        f"{format_value(group_value)} with {format_value(metric_value)} {metric_label}."
    )
