from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import pandas as pd

from historyAssistant.execution.query_models import QueryPlan
from historyAssistant.execution.strategies.base import AggregationStrategy, es_field_name


class RoomSwitchWithinMinutesStrategy(AggregationStrategy):
    needs_raw_values = True

    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        return {}, True

    def resolve_limit(self, plan: QueryPlan, metric_cfg: dict, default_limit: int, max_result_rows: int) -> Optional[int]:
        return int(plan.limit or metric_cfg.get("default_result_limit") or default_limit)

    def _query_with_required_workflow(self, query_filter: dict, metric_cfg: dict, plan: QueryPlan) -> dict:
        query = deepcopy(query_filter) if query_filter else {"match_all": {}}
        workflow_name = metric_cfg.get("required_workflow_name")
        workflow_field = metric_cfg.get("workflow_name_field", "workflow_name")
        if not workflow_name:
            return query
        required_filter = {"term": {es_field_name(workflow_field, plan): workflow_name}}
        if "bool" not in query:
            return {"bool": {"filter": [query, required_filter]}}
        filters = query.setdefault("bool", {}).setdefault("filter", [])
        filters.append(required_filter)
        return query

    def build_query_body(
        self,
        query_filter: dict,
        metric_cfg: dict,
        group_cols: list,
        max_result_rows: int,
        ranking_order: dict,
        plan: QueryPlan,
        pre_sort_fetch_size: Optional[int] = None,
    ) -> dict:
        person_field = metric_cfg.get("person_field", "initiator_person")
        timestamp_field = metric_cfg.get("timestamp_field", "timestamp_utc")
        scan_size = int(metric_cfg.get("max_scan_rows") or pre_sort_fetch_size or max_result_rows)
        source_fields = [
            "workflow_id",
            "workflow_date",
            timestamp_field,
            "shift",
            person_field,
            metric_cfg.get("workflow_name_field", "workflow_name"),
            "ward_name",
            metric_cfg.get("room_field", "room_name"),
        ]
        return {
            "size": scan_size,
            "query": self._query_with_required_workflow(query_filter, metric_cfg, plan),
            "sort": [
                {es_field_name(person_field, plan): {"order": "asc"}},
                {timestamp_field: {"order": "asc"}},
            ],
            "_source": source_fields,
            "track_total_hits": True,
        }

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        hits = es_response.get("hits", {}).get("hits", [])
        records = [hit.get("_source", {}) for hit in hits]
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        person_field = metric_cfg.get("person_field", "initiator_person")
        room_field = metric_cfg.get("room_field", "room_name")
        timestamp_field = metric_cfg.get("timestamp_field", "timestamp_utc")
        window_minutes = float(metric_cfg.get("window_minutes") or 5)
        limit = int(plan.limit or metric_cfg.get("default_result_limit") or 20)

        required = {person_field, room_field, timestamp_field}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        df = df.dropna(subset=[person_field, room_field, timestamp_field]).copy()
        df = df[(df[person_field] != "") & (df[room_field] != "")]
        if df.empty:
            return pd.DataFrame()

        df["_timestamp_dt"] = pd.to_datetime(df[timestamp_field], errors="coerce", utc=True)
        df = df.dropna(subset=["_timestamp_dt"])
        df = df.sort_values([person_field, "_timestamp_dt"], kind="mergesort")

        findings: List[Dict] = []
        for person, group in df.groupby(person_field, sort=False):
            rows = group.to_dict(orient="records")
            start = 0
            for idx, current in enumerate(rows):
                current_ts = current["_timestamp_dt"]
                while start < idx:
                    delta_from_start = (current_ts - rows[start]["_timestamp_dt"]).total_seconds() / 60.0
                    if delta_from_start <= window_minutes:
                        break
                    start += 1
                for previous in rows[start:idx]:
                    if str(previous.get(room_field)) == str(current.get(room_field)):
                        continue
                    minutes_between = (current_ts - previous["_timestamp_dt"]).total_seconds() / 60.0
                    if minutes_between < 0 or minutes_between > window_minutes:
                        continue
                    findings.append({
                        "initiator_person": person,
                        "first_room": previous.get(room_field),
                        "second_room": current.get(room_field),
                        "first_timestamp": previous.get(timestamp_field),
                        "second_timestamp": current.get(timestamp_field),
                        "minutes_between": round(minutes_between, 2),
                        "first_workflow_id": previous.get("workflow_id"),
                        "second_workflow_id": current.get("workflow_id"),
                        "first_ward": previous.get("ward_name"),
                        "second_ward": current.get("ward_name"),
                    })
                    if len(findings) >= limit:
                        return pd.DataFrame(findings)
        return pd.DataFrame(findings)


__all__ = ["RoomSwitchWithinMinutesStrategy"]
