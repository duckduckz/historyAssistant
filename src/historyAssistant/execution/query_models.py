from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from historyAssistant.config import ConfigBundle, get_runtime_defaults, load_all_configs
from historyAssistant.planning import REQUEST_TYPE_CHART, REQUEST_TYPE_QA

YamlDict = Dict[str, Any]

CANONICAL_TO_PHYSICAL_TABLE = {
    "historyCalls": "fact_calls",
    "historyLocations": "fact_locations",
    "historyWorkflows": "fact_workflows",
}


@dataclass(frozen=True)
class CompareToRange:
    date_rule: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    filters: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class QueryFilter:
    field: str
    op: str
    value: Any
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> YamlDict:
        return {
            "field": self.field,
            "op": self.op,
            "value": self.value,
            "meta": dict(self.meta) if self.meta else None,
        }

@dataclass(frozen=True)
class QueryPlan:
    metric: str
    request_type: str = REQUEST_TYPE_QA
    target_table: str = "fact_calls"
    canonical_table: str = "historyCalls"
    aggregation: Optional[str] = None
    metric_params: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    keyword_fields: List[str] = field(default_factory=list)
    filters: List[QueryFilter] = field(default_factory=list)
    ranking_order: Optional[Dict[str, Any]] = None
    display_order: List[Dict[str, Any]] = field(default_factory=list)
    order_by: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    scope_policy: str = "qa_ready"
    outlier_policy: str = "exclude_negative"
    export_all: bool = False
    source_fields: List[str] = field(default_factory=list)
    chart_spec: Optional[Dict[str, Any]] = None
    normalized_sub_request: Optional[str] = None
    sub_request_id: Optional[int] = None
    classification_confidence: Optional[float] = None
    valid: bool = True
    unsupported_reason: Optional[str] = None
    compare_to: Optional[CompareToRange] = None
    bucket_interval: Optional[str] = None
    bucket_summary: Optional[str] = None

    def effective_ranking_order(self) -> Optional[Dict[str, Any]]:
        return self.ranking_order or self.order_by

    def effective_order_by(self) -> Optional[Dict[str, Any]]:
        return self.order_by or self.ranking_order

    def to_dict(self) -> YamlDict:
        return {
            "metric": self.metric,
            "request_type": self.request_type,
            "target_table": self.target_table,
            "canonical_table": self.canonical_table,
            "aggregation": self.aggregation,
            "metric_params": dict(self.metric_params),
            "group_by": list(self.group_by),
            "keyword_fields": list(self.keyword_fields),
            "filters": [item.to_dict() for item in self.filters],
            "ranking_order": dict(self.ranking_order) if self.ranking_order else None,
            "display_order": [dict(item) for item in self.display_order],
            "order_by": dict(self.order_by) if self.order_by else None,
            "limit": self.limit,
            "scope_policy": self.scope_policy,
            "outlier_policy": self.outlier_policy,
            "export_all": self.export_all,
            "source_fields": list(self.source_fields),
            "chart_spec": dict(self.chart_spec) if self.chart_spec else None,
            "normalized_sub_request": self.normalized_sub_request,
            "sub_request_id": self.sub_request_id,
            "classification_confidence": self.classification_confidence,
            "valid": self.valid,
            "unsupported_reason": self.unsupported_reason,
            "compare_to": {
                "date_rule": self.compare_to.date_rule,
                "start_date": self.compare_to.start_date,
                "end_date": self.compare_to.end_date,
                "filters": list(self.compare_to.filters),
            } if self.compare_to else None,
            "bucket_interval": self.bucket_interval,
            "bucket_summary": self.bucket_summary,
        }


__all__ = [
    "CANONICAL_TO_PHYSICAL_TABLE",
    "QueryFilter",
    "QueryPlan",
    "CompareToRange",
    "apply_numeric_summary_metric_params",
    "infer_scope_and_outlier",
    "dict_filter_to_query_filter",
    "canonical_plan_to_query_plan",
]


def infer_scope_and_outlier(metric: str,*,metric_catalog: YamlDict,runtime_defaults: YamlDict,scope_policy: Optional[str] = None,outlier_policy: Optional[str] = None) -> Tuple[str, str]:
    metric_cfg = metric_catalog.get("metrics", {}).get(metric, {})
    inferred_scope = scope_policy or metric_cfg.get("default_scope") or runtime_defaults.get("scope_policy", "qa_ready")
    inferred_outlier = (
        outlier_policy
        or metric_cfg.get("default_outlier_policy")
        or runtime_defaults.get("outlier_policy", "exclude_negative")
    )

    return str(inferred_scope), str(inferred_outlier)


def apply_numeric_summary_metric_params(metric_cfg: YamlDict, metric_params: YamlDict) -> YamlDict:
    if metric_cfg.get("aggregation") != "numeric_summary":
        return metric_cfg
    updated = dict(metric_cfg)
    field = metric_params.get("field")
    numeric_aggregation = metric_params.get("numeric_aggregation") or metric_params.get("aggregation")
    if field:
        updated["field"] = field
        updated["required_columns"] = [field]
    if numeric_aggregation:
        updated["numeric_aggregation"] = str(numeric_aggregation).strip().lower()
    if metric_params.get("percentile") is not None:
        updated["percentile"] = metric_params.get("percentile")
    if metric_params.get("unit") is not None:
        updated["unit"] = metric_params.get("unit")
    if metric_params.get("field_label") is not None:
        updated["field_label"] = metric_params.get("field_label")
    return updated


def dict_filter_to_query_filter(filter_dict: Dict[str, Any]) -> QueryFilter:
    return QueryFilter(
        field=filter_dict.get("field"),
        op=filter_dict.get("op"),
        value=filter_dict.get("value"),
        meta=filter_dict.get("meta"),
    )


def _active_table_definition(bundle: ConfigBundle, canonical_table: str) -> YamlDict:
    tables = bundle.table_registry.get("tables", {}) or {}
    return tables.get(canonical_table, {}) or {}


def _date_field_for_table(bundle: ConfigBundle, canonical_table: str) -> str:
    return str(_active_table_definition(bundle, canonical_table).get("date_field") or "call_date")


def _timestamp_field_for_table(bundle: ConfigBundle, canonical_table: str) -> str:
    return str(_active_table_definition(bundle, canonical_table).get("timestamp_field") or "timestamp_utc")


def _has_date_filter(filters: List[QueryFilter], date_field: str) -> bool:
    date_ops = {"relative", "between", "=", ">=", ">", "<=", "<"}
    return any(
        filter_item.field == date_field and filter_item.op in date_ops
        for filter_item in filters
    )


def canonical_plan_to_query_plan(plan: Dict[str, Any],*,config_bundle: Optional[ConfigBundle] = None) -> QueryPlan:
    bundle = config_bundle or load_all_configs()
    runtime_defaults = get_runtime_defaults(bundle.app_config)
    metric_catalog = bundle.metric_catalog

    metric = plan.get("metric") or "total_calls"
    target_tables = plan.get("target_tables", []) or []
    if not isinstance(target_tables, list):
        raise ValueError("canonical plan field 'target_tables' must be a list")
    if not target_tables:
        default_table = next(iter((bundle.table_registry.get("tables", {}) or {}).keys()), "historyCalls")
        target_tables = [default_table]
    if len(target_tables) != 1:
        raise ValueError(f"execution supports exactly one target table, got: {target_tables}")

    canonical_table = target_tables[0]
    physical_table = CANONICAL_TO_PHYSICAL_TABLE.get(canonical_table)
    if not physical_table:
        raise ValueError(f"unsupported canonical target table: {canonical_table}")
    date_field = _date_field_for_table(bundle, canonical_table)
    timestamp_field = _timestamp_field_for_table(bundle, canonical_table)
    table_def = _active_table_definition(bundle, canonical_table)

    request_type = plan.get("request_type", REQUEST_TYPE_QA)
    if request_type not in {REQUEST_TYPE_QA, REQUEST_TYPE_CHART}:
        raise ValueError(f"unsupported request_type for execution adapter: {request_type}")

    metric_cfg = dict((metric_catalog.get("metrics", {}) or {}).get(metric, {}) or {})
    metric_params = dict(plan.get("metric_params") or {}) if isinstance(plan.get("metric_params"), dict) else {}
    metric_cfg = apply_numeric_summary_metric_params(metric_cfg, metric_params)

    resolved_scope, resolved_outlier = infer_scope_and_outlier(
        metric,
        metric_catalog={"metrics": {metric: metric_cfg}},
        runtime_defaults=runtime_defaults,
        scope_policy=plan.get("scope_policy"),
        outlier_policy=plan.get("outlier_policy"),
    )

    compare_to_data = plan.get("compare_to")
    compare_to_range = None
    if compare_to_data:
        date_range = compare_to_data.get("date_range") or {}
        compare_to_range = CompareToRange(
            date_rule=compare_to_data.get("date_rule"),
            start_date=date_range.get("start_date"),
            end_date=date_range.get("end_date"),
            filters=compare_to_data.get("filters") or [],
        )

    execution_filters = [dict_filter_to_query_filter(item) for item in plan.get("filters", [])]

    date_rule = plan.get("date_rule")
    date_range = plan.get("date_range")
    date_params = plan.get("date_params")

    if date_rule and date_rule != "none" and not _has_date_filter(execution_filters, date_field):
        if date_rule == "custom_range" and date_range:
            start_date = date_range.get("start_date")
            end_date = date_range.get("end_date")
            if start_date and end_date:
                execution_filters.append(QueryFilter(
                    field=date_field,
                    op="between",
                    value=[start_date, end_date],
                ))
        elif date_rule == "rolling_days" and date_params:
            days = date_params.get("days")
            if days:
                execution_filters.append(QueryFilter(
                    field=date_field,
                    op="relative",
                    value=f"last_{days}_days",
                ))
        else:
            execution_filters.append(QueryFilter(
                field=date_field,
                op="relative",
                value=date_rule,
            ))

    time_range = plan.get("time_range")
    if time_range and time_range.get("start_time") and time_range.get("end_time"):
        execution_filters.append(QueryFilter(
            field=timestamp_field,
            op="time_between",
            value=(time_range["start_time"], time_range["end_time"]),
        ))

    return QueryPlan(
        metric=metric,
        request_type=request_type,
        target_table=physical_table,
        aggregation=plan.get("aggregation") or metric_cfg.get("aggregation"),
        metric_params=metric_params,
        group_by=list(plan.get("group_by", []) or []),
        keyword_fields=list(table_def.get("keyword_fields", []) or []),
        filters=execution_filters,
        ranking_order=plan.get("ranking_order"),
        display_order=list(plan.get("display_order", []) or []),
        order_by=plan.get("order_by"),
        limit=plan.get("limit"),
        scope_policy=resolved_scope,
        outlier_policy=resolved_outlier,
        export_all=bool(plan.get("export_all", False)),
        source_fields=list(plan.get("source_fields", []) or []),
        chart_spec=plan.get("chart_spec"),
        normalized_sub_request=plan.get("normalized_sub_request"),
        sub_request_id=plan.get("sub_request_id"),
        classification_confidence=plan.get("classification_confidence"),
        valid=bool(plan.get("valid", True)),
        unsupported_reason=plan.get("unsupported_reason"),
        compare_to=compare_to_range,
        bucket_interval=plan.get("bucket_interval"),
        bucket_summary=plan.get("bucket_summary"),
        canonical_table=canonical_table,

    )
