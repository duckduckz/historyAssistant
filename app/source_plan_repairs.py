from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

YamlDict = Dict[str, Any]

from historyAssistant.planning.intent_planner.plan_builder import build_plan_from_slots
from historyAssistant.planning.intent_planner.record_intent import extract_limit, extract_record_intent_slots

from historyAssistant.planning.extraction.ranking_signals import detect_ranking_signals

_HASHED_PERSON_RE = re.compile(r"[a-fA-F0-9]{64}(?![a-fA-F0-9])")
_COMPARE_RE = re.compile(r"(?i)\b(?:compare|comparison|versus|vs\.?|between)\b")
_LAST_LOCATION_RE = re.compile(
    r"(?i)\b(?:last seen|last known location|latest location|most recent location|current location|where is|where was|last moved to|last moved)\b"
)
_LOCATION_MOVEMENT_RE = re.compile(
    r"(?i)\b(?:exercise|exercised|exercising|walk|walked|walking|movement|moved|mobility|activity)\b"
)
_WORKFLOW_PRESENCE_LOOKUP_RE = re.compile(
    r"(?i)\b(?:who was present|who is present|show presence|list presence|presence records|nurse present|staff present)\b"
)
_WORKFLOW_ROOM_SWITCH_RE = re.compile(
    r"(?i)\b(?:nurse|staff|person|initiator)?\s*(?:present|presence|in)\b.*\b(?:two|2|different)(?:\s+different)?\s+(?:rooms?|room_name)\b.*\bwithin\s+(\d+)\s+minutes?\b"
)


def repair_raw_records_sort(
    plan: YamlDict,
    *,
    question: str,
    bundle: Any,
    detected_entities: Optional[YamlDict] = None,
) -> YamlDict:
    
    if plan.get("aggregation") != "raw_records":
        return plan
    if plan.get("order_by") or plan.get("ranking_order"):
        return plan

    ranking = detect_ranking_signals(question)
    if not ranking.detected:
        return plan

    metric_catalog = (bundle.metric_catalog or {}).get("metrics") or {}
    sort_field: Optional[str] = None
    direction: Optional[str] = None
    question_lower = (question or "").lower()

    # 1. Scan metric catalog sort_aliases — config-driven phrase → field mapping
    for metric_cfg in metric_catalog.values():
        if not isinstance(metric_cfg, dict):
            continue
        sort_aliases = metric_cfg.get("sort_aliases") or {}
        if not isinstance(sort_aliases, dict):
            continue
        field = metric_cfg.get("field")
        if not field:
            continue
        for dir_key in ("desc", "asc"):
            phrases = sort_aliases.get(dir_key) or []
            if any(str(p).lower() in question_lower for p in phrases):
                sort_field = field
                direction = dir_key
                break
        if sort_field:
            break

    # 2. Fall back to detected metric entity (e.g. "sorted by response time")
    if not sort_field:
        metrics_detected = (detected_entities or {}).get("metrics") or []
        for m in metrics_detected:
            field = (metric_catalog.get(m) or {}).get("field")
            if field:
                sort_field = field
                direction = ranking.direction or "desc"
                break

    if not sort_field:
        return plan

    updated = dict(plan)
    order = {"field": sort_field, "direction": direction or "desc"}
    updated["order_by"] = order
    updated["ranking_order"] = order
    if not updated.get("limit"):
        updated["limit"] = extract_limit(question) or 10
    return updated


def active_table_definition(bundle: Any) -> YamlDict:
    tables = bundle.table_registry.get("tables", {}) or {}
    active_table = next(iter(tables.keys()), None)
    if not active_table:
        active_table = (
            bundle.app_config.get("tables", {}).get("active_table")
            or bundle.chart_catalog.get("policy", {}).get("active_table")
            or "historyCalls"
        )
    return tables.get(active_table, {}) or {}


def active_date_field(bundle: Any) -> str:
    return str(active_table_definition(bundle).get("date_field") or "call_date")


def active_timestamp_field(bundle: Any) -> str:
    table_def = active_table_definition(bundle)
    return str(table_def.get("timestamp_field") or "timestamp_utc")


def _normalize_location_area_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip().upper()
    compact = re.fullmatch(r"([0-9][A-Z])[\s_]+([A-Z]+)[\s_]*(\d{1,3})", text)
    if compact:
        ward_prefix, location_type, digits = compact.groups()
        return f"{ward_prefix}_{location_type} {digits.zfill(3)}"
    return value


def _repair_location_filters(filters: list[YamlDict]) -> list[YamlDict]:
    repaired_filters = []
    for filt in filters:
        repaired = dict(filt)
        field = repaired.get("field")
        if field == "room_name":
            repaired["field"] = "area"
            field = "area"
        if field in {"area", "prev_location"}:
            repaired["value"] = _normalize_location_area_value(repaired.get("value"))
        repaired_filters.append(repaired)
    return repaired_filters


def repair_location_person_lookup(canonical_plan: YamlDict, *, question: str, bundle: Any) -> YamlDict:
    tables = bundle.table_registry.get("tables", {}) or {}
    active_table = next(iter(tables.keys()), "")
    if active_table != "historyLocations":
        return canonical_plan

    updated = dict(canonical_plan)
    filters = _repair_location_filters(list(updated.get("filters") or []))

    raw_hashes = _HASHED_PERSON_RE.findall(question or "")
    unique_hashes: List[str] = []
    for raw in raw_hashes:
        lowered = raw.lower()
        if lowered not in unique_hashes:
            unique_hashes.append(lowered)

    is_compare = bool(_COMPARE_RE.search(question or ""))

    if unique_hashes:
        filters = [
            f for f in filters
            if f.get("field") not in {"personId", "personFirstName"}
        ]

        if is_compare and len(unique_hashes) >= 3:
            filters.append({"field": "personFirstName", "op": "in", "value": list(unique_hashes)})
            existing_group_by = list(updated.get("group_by") or [])
            if "personFirstName" not in existing_group_by:
                existing_group_by.insert(0, "personFirstName")
            updated["group_by"] = existing_group_by
            updated["ranking_order"] = {
                "field": "metric_value",
                "direction": "desc",
                "target_group": "personFirstName",
            }
            updated["compare_to"] = None
        elif is_compare and len(unique_hashes) == 2:
            filters.append({"field": "personFirstName", "op": "=", "value": unique_hashes[0]})
            existing_compare = updated.get("compare_to") or {}
            compare_filters = [
                f for f in (existing_compare.get("filters") or [])
                if f.get("field") not in {"personId", "personFirstName"}
            ]
            compare_filters.append(
                {"field": "personFirstName", "op": "=", "value": unique_hashes[1]}
            )
            updated["compare_to"] = {
                **existing_compare,
                "filters": compare_filters,
            }
        else:
            filters.append({"field": "personFirstName", "op": "=", "value": unique_hashes[0]})

    updated["filters"] = filters
    updated["target_tables"] = ["historyLocations"]

    valid_location_metrics = {
        "location_update_count", "movement_event_count", "movement_details",
        "last_known_location", "location_details",
    }
    llm_metric = updated.get("metric", "")
    llm_chose_valid = llm_metric in valid_location_metrics

    single_person = len(unique_hashes) == 1 and not is_compare

    if single_person and _LAST_LOCATION_RE.search(question or ""):
        updated["metric"] = "last_known_location"
        updated["aggregation"] = "raw_records"
        updated["group_by"] = []
        updated["ranking_order"] = None
        updated["order_by"] = {"field": "timestamp", "direction": "desc"}
        updated["limit"] = 1
    elif not llm_chose_valid and _LOCATION_MOVEMENT_RE.search(question or ""):
        updated["metric"] = "movement_event_count"
        updated["aggregation"] = "cardinality"
        if "group_by" not in updated:
            updated["group_by"] = []

    return updated


def should_apply_picker_date_fallback(canonical_plan: YamlDict) -> bool:
    return canonical_plan.get("metric") != "last_known_location"


def _normalize_workflow_room_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().upper()
    simple = re.fullmatch(r"(K|SANITAIR)\s*(\d{1,3})", text)
    if simple:
        prefix, digits = simple.groups()
        return f"{prefix} {digits.zfill(3)}"
    dvc = re.fullmatch(r"(DVC_[A-Z]+)\s*(\d{1,3})", text)
    if dvc:
        prefix, digits = dvc.groups()
        return f"{prefix} {digits.zfill(3)}"
    return value


def repair_workflow_presence_lookup(canonical_plan: YamlDict, *, question: str, bundle: Any) -> YamlDict:
    tables = bundle.table_registry.get("tables", {}) or {}
    active_table = next(iter(tables.keys()), "")
    if active_table != "historyWorkflows":
        return canonical_plan

    metric_catalog = bundle.metric_catalog.get("metrics", {}) or {}
    metric_name = canonical_plan.get("metric") or ""

    required_workflow_name: Optional[str] = (
        metric_catalog.get(metric_name, {}).get("required_workflow_name")
    )
    if not required_workflow_name and canonical_plan.get("aggregation") == "raw_records":
        for m_key, m_cfg in metric_catalog.items():
            if m_cfg.get("aggregation") == "raw_records" and m_cfg.get("required_workflow_name"):
                required_workflow_name = m_cfg["required_workflow_name"]
                metric_name = m_key
                break

    updated = dict(canonical_plan)
    filters = list(updated.get("filters") or [])
    repaired_filters = []
    has_workflow_filter = False

    for filt in filters:
        repaired = dict(filt)
        if repaired.get("field") == "room_name":
            repaired["value"] = _normalize_workflow_room_value(repaired.get("value"))
        if repaired.get("field") == "workflow_name" and required_workflow_name:
            if str(repaired.get("value")) == required_workflow_name:
                has_workflow_filter = True
        repaired_filters.append(repaired)

    if required_workflow_name and not has_workflow_filter:
        repaired_filters.append({"field": "workflow_name", "op": "=", "value": required_workflow_name})

    if updated.get("aggregation") == "raw_records" and metric_name != "presence_room_switch_within_minutes":
        updated["metric"] = metric_name
        updated["aggregation"] = "raw_records"
        updated["group_by"] = []
        updated["ranking_order"] = None
        updated["order_by"] = {"field": active_timestamp_field(bundle), "direction": "desc"}

    raw_hashes = _HASHED_PERSON_RE.findall(question or "")
    unique_hashes: List[str] = []
    for raw in raw_hashes:
        lowered = raw.lower()
        if lowered not in unique_hashes:
            unique_hashes.append(lowered)

    is_compare = bool(_COMPARE_RE.search(question or ""))

    if unique_hashes:
        repaired_filters = [
            f for f in repaired_filters
            if f.get("field") != "initiator_person"
        ]

        if is_compare and len(unique_hashes) >= 3:
            repaired_filters.append(
                {"field": "initiator_person", "op": "in", "value": list(unique_hashes)}
            )
            existing_group_by = list(updated.get("group_by") or [])
            if "initiator_person" not in existing_group_by:
                existing_group_by.insert(0, "initiator_person")
            updated["group_by"] = existing_group_by
            updated["ranking_order"] = {
                "field": "metric_value",
                "direction": "desc",
                "target_group": "initiator_person",
            }
            updated["compare_to"] = None
            if metric_name not in {"presence_event_count", "presence_room_switch_within_minutes"}:
                updated["metric"] = "presence_event_count"
                updated["aggregation"] = "count_threshold"
        elif is_compare and len(unique_hashes) == 2:
            repaired_filters.append(
                {"field": "initiator_person", "op": "=", "value": unique_hashes[0]}
            )
            existing_compare = updated.get("compare_to") or {}
            compare_filters = [
                f for f in (existing_compare.get("filters") or [])
                if f.get("field") != "initiator_person"
            ]
            compare_filters.append(
                {"field": "initiator_person", "op": "=", "value": unique_hashes[1]}
            )
            updated["compare_to"] = {
                **existing_compare,
                "filters": compare_filters,
            }
            if metric_name not in {"presence_event_count", "presence_room_switch_within_minutes"}:
                updated["metric"] = "presence_event_count"
                updated["aggregation"] = "count_threshold"
        elif len(unique_hashes) == 1:
            repaired_filters.append(
                {"field": "initiator_person", "op": "=", "value": unique_hashes[0]}
            )

    updated["filters"] = repaired_filters
    updated["target_tables"] = ["historyWorkflows"]
    return updated


def build_workflow_room_switch_plan(item: YamlDict, *, bundle: Any) -> Optional[YamlDict]:
    tables = bundle.table_registry.get("tables", {}) or {}
    active_table = next(iter(tables.keys()), "")
    if active_table != "historyWorkflows":
        return None

    question = str(item.get("text") or "")
    match = _WORKFLOW_ROOM_SWITCH_RE.search(question)
    if not match:
        return None

    window_minutes = int(match.group(1))
    metric_catalog = bundle.metric_catalog.get("metrics", {}) or {}
    metric_cfg = metric_catalog.get("presence_room_switch_within_minutes", {}) or {}
    default_window = metric_cfg.get("window_minutes")

    plan: YamlDict = {
        "request_type": item.get("request_type", "qa"),
        "target_tables": ["historyWorkflows"],
        "normalized_sub_request": question,
        "sub_request_id": item.get("sub_request_id"),
        "classification_confidence": 1.0,
        "metric": "presence_room_switch_within_minutes",
        "aggregation": "room_switch_within_minutes",
        "filters": [],
        "group_by": [],
        "ranking_order": None,
        "order_by": {"field": active_timestamp_field(bundle), "direction": "asc"},
        "limit": int(metric_cfg.get("default_result_limit") or 20),
        "scope_policy": "raw",
        "outlier_policy": "none",
        "date_rule": "none",
        "date_range": None,
    }
    if default_window != window_minutes:
        plan["_requested_window_minutes"] = window_minutes
    return plan


def build_workflow_presence_lookup_plan(
    item: YamlDict,
    *,
    bundle: Any,
    detected_entities: Optional[YamlDict] = None,
) -> Optional[YamlDict]:
    tables = bundle.table_registry.get("tables", {}) or {}
    active_table = next(iter(tables.keys()), "")
    if active_table != "historyWorkflows":
        return None
    question = str(item.get("text") or "")
    if not _WORKFLOW_PRESENCE_LOOKUP_RE.search(question):
        return None

    extraction = extract_record_intent_slots(
        question=question,
        normalized_question=question,
        source_id="workflows",
        target_table="historyWorkflows",
        detected_entities=detected_entities,
        request_type=item.get("request_type", "qa"),
        bundle=bundle,
    )
    if not extraction.used or not extraction.slots:
        return None
    plan = build_plan_from_slots(extraction.slots, bundle=bundle)
    plan["sub_request_id"] = item.get("sub_request_id")
    plan["request_type"] = item.get("request_type", plan.get("request_type", "qa"))
    return repair_workflow_presence_lookup(plan, question=question, bundle=bundle)
