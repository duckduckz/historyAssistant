from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from historyAssistant.config import YamlDict, load_source_configs, source_id_for_table

EXPORT_DIR = Path(__file__).resolve().parents[3] / "exports"


def _new_export_path(suffix: str) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    import uuid
    return EXPORT_DIR / f"historyassistant_{uuid.uuid4().hex}{suffix}"


def build_export_file(execution_items: List[YamlDict], export_mode: str) -> Optional[str]:
    if export_mode == "No export":
        return None

    if export_mode == "Export answers":
        payload = {
            "results": [
                {
                    "sub_request_id": item.get("sub_request_id"),
                    "sub_question": item.get("sub_question"),
                    "request_type": item.get("request_type"),
                    "query": item.get("execution_result", {}).get("query"),
                    "records": item.get("execution_result", {}).get("records", []),
                    "scalar_answer": item.get("execution_result", {}).get("scalar_answer"),
                }
                for item in execution_items
            ]
        }
        path = _new_export_path(".json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return str(path)

    rows: List[YamlDict] = []
    for item in execution_items:
        execution_result = item.get("execution_result", {})
        records = execution_result.get("records", [])
        if records:
            for row in records:
                flattened = {
                    "sub_request_id": item.get("sub_request_id"),
                    "sub_question": item.get("sub_question"),
                    "request_type": item.get("request_type"),
                }
                flattened.update(row)
                rows.append(flattened)
            continue
        rows.append({
            "sub_request_id": item.get("sub_request_id"),
            "sub_question": item.get("sub_question"),
            "request_type": item.get("request_type"),
            "scalar_answer": execution_result.get("scalar_answer"),
        })

    frame = pd.DataFrame(rows)
    path = _new_export_path(".csv")
    frame.to_csv(path, index=False)
    return str(path)


def preview_dataframe(execution_items: List[YamlDict]) -> pd.DataFrame:
    rows: List[YamlDict] = []
    for item in execution_items:
        execution_result = item.get("execution_result", {})
        records = execution_result.get("records", [])[:70]
        if records:
            for row in records:
                rows.append(dict(row))
        elif execution_result.get("scalar_answer") is not None:
            rows.append({"scalar_answer": execution_result.get("scalar_answer")})
    return pd.DataFrame(rows)


def enrich_raw_record_summary_if_needed(question: str, execution_result: YamlDict, *, es: Any, bundle: Any) -> YamlDict:
    from historyAssistant.app.narration import needs_full_result_download, raw_record_summary_answer
    from historyAssistant.execution.executor import execute_full_raw_records_export

    if raw_record_summary_answer(question, execution_result) is None:
        return execution_result
    if not needs_full_result_download(execution_result):
        return execution_result

    export_plan = deepcopy(execution_result.get("canonical_plan") or {})
    total_count = int(execution_result.get("total_count", 0) or 0)
    if not export_plan or total_count <= 0:
        return execution_result

    export_plan["limit"] = min(total_count, 1000)
    export_plan["export_all"] = True
    full_result = execute_full_raw_records_export(export_plan, es=es, bundle=bundle)
    if full_result.get("ok") and full_result.get("records"):
        return full_result
    return execution_result


def build_export_execution_items(
    execution_items: List[YamlDict],
    *,
    export_mode: str,
    es: Any,
    bundle: Any,
) -> tuple[List[YamlDict], str]:
    from historyAssistant.app.narration import needs_full_result_download
    from historyAssistant.execution.executor import execute_full_raw_records_export

    auto_export = any(
        needs_full_result_download(item.get("execution_result", {}) or {})
        for item in execution_items
    )
    effective_mode = export_mode if export_mode != "No export" else ("Export table" if auto_export else "No export")
    if effective_mode == "No export":
        return execution_items, effective_mode

    export_items = deepcopy(execution_items)
    for item in export_items:
        execution_result = item.get("execution_result", {}) or {}
        if not needs_full_result_download(execution_result):
            continue
        export_plan = deepcopy(execution_result.get("canonical_plan") or {})
        total_count = int(execution_result.get("total_count", 0) or 0)
        if not export_plan or total_count <= 0:
            continue
        export_plan["limit"] = total_count
        export_plan["export_all"] = True
        runtime_source_id = source_id_for_table((export_plan.get("target_tables") or [None])[0])
        runtime_bundle = load_source_configs(runtime_source_id) if runtime_source_id else bundle
        full_result = execute_full_raw_records_export(export_plan, es=es, bundle=runtime_bundle)
        if full_result.get("ok") and full_result.get("records"):
            item["execution_result"] = full_result

    return export_items, effective_mode


def prepare_first_execution_result(
    execution_items: List[YamlDict],
    *,
    es: Any,
    bundle: Any,
) -> None:
    if not execution_items:
        return
    first = execution_items[0]
    execution_result = first.get("execution_result", {}) or {}
    canonical_plan = execution_result.get("canonical_plan") or {}
    runtime_source_id = source_id_for_table((canonical_plan.get("target_tables") or [None])[0])
    runtime_bundle = load_source_configs(runtime_source_id) if runtime_source_id else bundle
    first["execution_result"] = enrich_raw_record_summary_if_needed(
        first["sub_question"], execution_result, es=es, bundle=runtime_bundle,
    )
