from __future__ import annotations

from typing import Any, Dict

YamlDict = Dict[str, Any]


def get_narration_table_definition(execution_result: YamlDict) -> YamlDict:
    from historyAssistant.config import get_data_sources, load_all_configs, load_source_configs, source_id_for_table

    plan = execution_result.get("plan") or execution_result.get("canonical_plan") or {}
    target_tables = plan.get("target_tables") or []
    source_id = source_id_for_table(target_tables[0]) if target_tables else None
    if source_id:
        try:
            bundle = load_source_configs(source_id)
            for table_cfg in (bundle.table_registry or {}).get("tables", {}).values():
                return table_cfg or {}
        except Exception:
            pass

    records = execution_result.get("records") or []
    if not records:
        return {}

    record_fields = set().union(*(record.keys() for record in records[:5]))
    best_score = 0
    best_table_def: YamlDict = {}
    try:
        app_bundle = load_all_configs()
        for candidate_source_id in get_data_sources(app_bundle.app_config).keys():
            bundle = load_source_configs(str(candidate_source_id))
            for table_cfg in (bundle.table_registry or {}).get("tables", {}).values():
                candidate_fields = set(table_cfg.get("default_source_fields") or [])
                candidate_fields.update(table_cfg.get("dimensions") or [])
                candidate_fields.update(table_cfg.get("measure_fields") or [])
                candidate_fields.update(table_cfg.get("timestamp_fields") or [])
                candidate_fields.update((table_cfg.get("narration") or {}).get("compact_fields") or [])
                score = len(record_fields & candidate_fields)
                if score > best_score:
                    best_score = score
                    best_table_def = table_cfg or {}
    except Exception:
        pass
    if best_score:
        return best_table_def
    return {}


def get_narration_table_config(execution_result: YamlDict) -> YamlDict:
    return (get_narration_table_definition(execution_result).get("narration") or {})
