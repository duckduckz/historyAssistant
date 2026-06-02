from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from elasticsearch import Elasticsearch

from historyAssistant.config import ConfigBundle
from historyAssistant.execution.query_models import QueryPlan, canonical_plan_to_query_plan
from historyAssistant.execution.result import to_raw_result

YamlDict = Dict[str, Any]


def compute_change(main_df: pd.DataFrame, comp_df: pd.DataFrame, group_by: List[str]) -> YamlDict:
    if main_df.empty or comp_df.empty:
        return {"has_data": False}

    if "metric_value" not in main_df.columns or "metric_value" not in comp_df.columns:
        return {"has_data": False}

    if not group_by or len(main_df) == 1:
        main_val = main_df["metric_value"].iloc[0] if len(main_df) > 0 else None
        comp_val = comp_df["metric_value"].iloc[0] if len(comp_df) > 0 else None

        if main_val is None or comp_val is None:
            return {"has_data": False}

        try:
            main_val = float(main_val)
            comp_val = float(comp_val)
            absolute_change = main_val - comp_val
            percent_change = ((main_val - comp_val) / comp_val * 100) if comp_val != 0 else None
            return {
                "has_data": True,
                "main_value": main_val,
                "comparison_value": comp_val,
                "absolute_change": absolute_change,
                "percent_change": percent_change,
                "direction": "up" if absolute_change > 0 else ("down" if absolute_change < 0 else "same"),
            }
        except (ValueError, TypeError):
            return {"has_data": False}

    if group_by:
        try:
            main_keys = set(main_df[group_by[0]].dropna()) if group_by[0] in main_df.columns else set()
            comp_keys = set(comp_df[group_by[0]].dropna()) if group_by[0] in comp_df.columns else set()
            only_main = main_keys - comp_keys
            only_comp = comp_keys - main_keys
            overlap_ratio = len(main_keys & comp_keys) / max(len(main_keys | comp_keys), 1)

            if overlap_ratio < 0.5 and (len(only_main) + len(only_comp)) > 2:
                return {
                    "has_data": False,
                    "suppressed": True,
                    "reason": (
                        f"Group keys differ too much between sides "
                        f"({len(only_main)} only in main, {len(only_comp)} only in comparison, "
                        f"{overlap_ratio:.0%} overlap). Comparison would be misleading."
                    ),
                }

            merged = main_df.merge(comp_df, on=group_by, how="inner", suffixes=("_main", "_comp"))

            if merged.empty:
                return {"has_data": False, "reason": "No overlapping groups between main and comparison"}

            merged["absolute_change"] = merged["metric_value_main"] - merged["metric_value_comp"]
            safe_comp = merged["metric_value_comp"].replace(0, pd.NA)
            merged["percent_change"] = (
                (merged["metric_value_main"] - merged["metric_value_comp"]) / safe_comp * 100
            )

            result: YamlDict = {
                "has_data": True,
                "grouped_changes": merged.to_dict(orient="records"),
            }
            if only_main or only_comp:
                result["partial_overlap"] = True
                result["groups_only_in_main"] = sorted(str(k) for k in only_main)
                result["groups_only_in_comparison"] = sorted(str(k) for k in only_comp)
            return result
        except Exception:
            return {"has_data": False}

    return {"has_data": False}


def execute_comparison_query(canonical_plan: YamlDict,compare_to: YamlDict,es: Elasticsearch,bundle: ConfigBundle,main_result: YamlDict) -> Optional[YamlDict]:
    
    from historyAssistant.execution.executor.orchestrator import execute_plan

    comparison_plan = dict(canonical_plan)
    comparison_plan.pop("compare_to", None)

    compare_date_rule = compare_to.get("date_rule")
    compare_date_range = compare_to.get("date_range")
    compare_date_params = compare_to.get("date_params")
    if compare_date_rule:
        comparison_plan["date_rule"] = compare_date_rule
    if compare_date_range:
        comparison_plan["date_range"] = compare_date_range
    if compare_date_params is not None:
        comparison_plan["date_params"] = compare_date_params
    if compare_date_rule or compare_date_range:
        main_filters = comparison_plan.get("filters", [])
        canonical_table = (comparison_plan.get("target_tables") or ["historyCalls"])[0]
        table_def = (bundle.table_registry.get("tables", {}) or {}).get(canonical_table, {}) or {}
        date_field = table_def.get("date_field", "call_date")
        comparison_plan["filters"] = [f for f in main_filters if f.get("field") != date_field]
        if compare_date_rule != "custom_range":
            comparison_plan["date_range"] = compare_date_range if compare_date_range else None
        if compare_date_rule != "rolling_days":
            comparison_plan["date_params"] = compare_date_params if compare_date_params else None

    compare_filters = compare_to.get("filters") or []
    if compare_filters:
        compare_field_set = {f.get("field") for f in compare_filters}
        main_filters = comparison_plan.get("filters", [])
        kept_filters = [f for f in main_filters if f.get("field") not in compare_field_set]
        comparison_plan["filters"] = kept_filters + list(compare_filters)

    try:
        comp_query_plan = canonical_plan_to_query_plan(comparison_plan, config_bundle=bundle)
        comp_execution = execute_plan(comp_query_plan, es, config_bundle=bundle)
        comp_raw = to_raw_result(comp_execution, comp_query_plan, config_bundle=bundle)

        if not comp_raw.get("ok", False):
            return None

        main_df = main_result.get("dataframe", pd.DataFrame())
        comp_df = comp_raw.get("dataframe", pd.DataFrame())
        change_info = compute_change(main_df, comp_df, canonical_plan.get("group_by", []))

        return {
            "ok": True,
            "period": compare_to,
            "dataframe": comp_df,
            "records": comp_raw.get("records", []),
            "scalar_answer": comp_raw.get("scalar_answer"),
            "change": change_info,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


__all__ = [
    "compute_change",
    "execute_comparison_query",
]
