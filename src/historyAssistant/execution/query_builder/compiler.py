from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Dict, List, Optional

from historyAssistant.config import ConfigBundle, get_runtime_defaults, load_all_configs
from historyAssistant.execution.strategies import STRATEGY_REGISTRY
from historyAssistant.execution.query_builder.date_filters import get_date_bounds, get_max_call_date, get_max_date
from historyAssistant.execution.query_builder.field_resolver import keyword_fields_for_plan, table_definition_for_plan
from historyAssistant.execution.query_builder.filter_builder import build_filters
from historyAssistant.execution.query_models import QueryPlan, apply_numeric_summary_metric_params

YamlDict = Dict[str, Any]


@dataclass(frozen=True)
class CompiledEsQuery:
    query_body: Dict[str, Any]
    metric_cfg: YamlDict
    filter_debug: List[str]
    effective_limit: Optional[int]
    overfetch_debug: Optional[YamlDict] = None
    needs_raw_values: bool = False

    def to_dict(self) -> YamlDict:
        return {
            "query_body": self.query_body,
            "metric_cfg": dict(self.metric_cfg),
            "filter_debug": list(self.filter_debug),
            "effective_limit": self.effective_limit,
            "overfetch_debug": dict(self.overfetch_debug) if self.overfetch_debug else None,
            "needs_raw_values": self.needs_raw_values,
        }


def compile_es_query(
    plan: QueryPlan,
    *,
    config_bundle: Optional[ConfigBundle] = None,
    reference_date: Optional[date] = None,
) -> CompiledEsQuery:
    bundle = config_bundle or load_all_configs()
    plan = _with_table_metadata(plan, bundle)
    runtime_defaults = get_runtime_defaults(bundle.app_config)
    default_limit = int(runtime_defaults.get("default_limit", 10))
    max_result_rows = int(runtime_defaults.get("max_result_rows", 50))
    pre_sort_fetch_multiplier = int(runtime_defaults.get("pre_sort_fetch_multiplier", 10))
    pre_sort_fetch_size = max_result_rows * max(pre_sort_fetch_multiplier, 1)

    metric_cfg = _metric_config(plan, bundle)
    aggregation = "time_bucket_summary" if plan.bucket_interval else metric_cfg.get("aggregation", "count")
    strategy = STRATEGY_REGISTRY.get(aggregation)
    if not strategy:
        raise ValueError(f"No strategy for aggregation: {aggregation}")

    es_filters, debug_filters = build_filters(plan, metric_cfg, reference_date, config_bundle=bundle)
    query_filter = {"bool": {"filter": es_filters}} if es_filters else {"match_all": {}}
    ranking_order = plan.effective_ranking_order() or {}
    effective_limit = strategy.resolve_limit(plan, metric_cfg, default_limit, max_result_rows)
    query_body = strategy.build_query_body(
        query_filter,
        metric_cfg,
        plan.group_by or [],
        max_result_rows,
        ranking_order,
        plan,
        pre_sort_fetch_size=pre_sort_fetch_size,
    )
    if effective_limit is not None and aggregation == "raw_records":
        query_body["size"] = effective_limit

    return CompiledEsQuery(
        query_body=query_body,
        metric_cfg=metric_cfg,
        filter_debug=debug_filters,
        effective_limit=effective_limit,
        overfetch_debug=_overfetch_debug(
            plan=plan,
            metric_cfg=metric_cfg,
            aggregation=aggregation,
            ranking_order=ranking_order,
            pre_sort_fetch_multiplier=pre_sort_fetch_multiplier,
            pre_sort_fetch_size=pre_sort_fetch_size,
        ),
        needs_raw_values=strategy.needs_raw_values,
    )


def _metric_config(plan: QueryPlan, bundle: ConfigBundle) -> YamlDict:
    metric_cfg = dict((bundle.metric_catalog.get("metrics", {}) or {}).get(plan.metric, {}) or {})
    if not metric_cfg:
        raise ValueError(f"Unsupported metric: {plan.metric}")
    if plan.metric_params:
        metric_cfg.update(dict(plan.metric_params))
    metric_cfg = apply_numeric_summary_metric_params(metric_cfg, plan.metric_params)

    table_def = table_definition_for_plan(plan, bundle)
    metric_cfg["_date_field"] = table_def.get("date_field", "call_date")
    metric_cfg["_timestamp_field"] = table_def.get("timestamp_field", "timestamp_utc")
    return metric_cfg


def _with_table_metadata(plan: QueryPlan, bundle: ConfigBundle) -> QueryPlan:
    if plan.keyword_fields:
        return plan
    return replace(plan, keyword_fields=sorted(keyword_fields_for_plan(plan, bundle)))


def _overfetch_debug(
    *,
    plan: QueryPlan,
    metric_cfg: YamlDict,
    aggregation: str,
    ranking_order: YamlDict,
    pre_sort_fetch_multiplier: int,
    pre_sort_fetch_size: int,
) -> YamlDict:
    group_by = list(plan.group_by or [])
    activated = (
        bool(ranking_order)
        and bool(group_by)
        and bool(metric_cfg.get("needs_overfetch"))
        and aggregation != "time_bucket_summary"
    )
    return {
        "activated": activated,
        "pre_sort_fetch_multiplier": pre_sort_fetch_multiplier,
        "pre_sort_fetch_size": pre_sort_fetch_size if activated else None,
        "group_by": group_by,
        "ranking_order": dict(ranking_order) if isinstance(ranking_order, dict) else None,
        "aggregation": aggregation,
    }


__all__ = [
    "CompiledEsQuery",
    "build_filters",
    "compile_es_query",
    "get_date_bounds",
    "get_max_call_date",
    "get_max_date",
]
