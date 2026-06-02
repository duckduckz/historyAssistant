from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

YamlDict = Dict[str, Any]


def apply_entity_intersection(canonical_plan: YamlDict, raw_result: YamlDict) -> YamlDict:
    spec = canonical_plan.get("entity_intersection") or {}
    entity_field = spec.get("entity_field")
    condition_field = spec.get("condition_field")
    required_values = [str(value) for value in spec.get("required_values") or []]
    df = raw_result.get("dataframe", pd.DataFrame())

    if (
        df.empty
        or not entity_field
        or not condition_field
        or entity_field not in df.columns
        or condition_field not in df.columns
        or "metric_value" not in df.columns
        or not required_values
    ):
        raw_result["dataframe"] = pd.DataFrame()
        raw_result["records"] = []
        raw_result["record_count"] = 0
        raw_result["scalar_answer"] = 0
        raw_result["empty"] = True
        return raw_result

    required_set = {str(value) for value in required_values}
    rows: List[YamlDict] = []
    for entity, group in df.groupby(entity_field, dropna=True, sort=False):
        present = {
            str(value)
            for value in group[condition_field].dropna().tolist()
            if str(value) in required_set
        }
        if present != required_set:
            continue
        metric_value = pd.to_numeric(group["metric_value"], errors="coerce").fillna(0).sum()
        rows.append(
            {
                entity_field: entity,
                "matched_values": ", ".join(required_values),
                "matched_value_count": len(present),
                "metric_value": int(metric_value) if float(metric_value).is_integer() else float(metric_value),
            }
        )

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(
            ["metric_value", entity_field], ascending=[False, True]
        ).reset_index(drop=True)

    raw_result["dataframe"] = result_df
    raw_result["records"] = result_df.to_dict(orient="records")
    raw_result["sample_records"] = raw_result["records"][:10]
    raw_result["record_count"] = len(result_df)
    raw_result["total_count"] = len(result_df)
    raw_result["scalar_answer"] = len(result_df)
    raw_result["empty"] = result_df.empty
    raw_result["result_type"] = "aggregated"
    raw_result["chart_payload"] = None
    raw_result["query_shape"] = "entity_intersection"
    return raw_result


__all__ = ["apply_entity_intersection"]
