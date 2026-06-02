from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from historyAssistant.config import load_source_configs
from historyAssistant.execution import execute_canonical_plan, open_es_connection

logger = logging.getLogger(__name__)

YamlDict = Dict[str, Any]


@dataclass
class ExecutionResult:
    # one entry per validated plan that was successfully executed
    execution_items: List[YamlDict] = field(default_factory=list)
    # open ES connection stay alive so the caller can reuse it for export
    es: Any = None


class ExecutionAgent:
    def __init__(self, *, validated_items: List[YamlDict], bundle: Any) -> None:
        self.validated_items = validated_items
        self.bundle = bundle

    def run(self) -> ExecutionResult:
        es = open_es_connection(config_bundle=self.bundle)
        execution_items: List[YamlDict] = []

        for vi in self.validated_items:
            item = vi["item"]
            runtime_plan = vi["validation"]["canonical_plan"]
            runtime_bundle = _bundle_for_plan(runtime_plan, self.bundle)
            execution_result = execute_canonical_plan(
                runtime_plan,
                es,
                config_bundle=runtime_bundle,
            )
            execution_items.append(
                {
                    "sub_request_id": item["sub_request_id"],
                    "sub_question": item.get("display_text", item["text"]),
                    "request_type": item["request_type"],
                    "execution_result": execution_result,
                    "narration": "",
                }
            )

        return ExecutionResult(execution_items=execution_items, es=es)


def _bundle_for_plan(plan: YamlDict, fallback_bundle: Any) -> Any:
    target_table = ((plan or {}).get("target_tables") or [None])[0]
    if not target_table:
        return fallback_bundle
    if target_table in ((fallback_bundle.table_registry.get("tables", {}) or {})):
        return fallback_bundle
    table_to_source = {
        "historyCalls": "calls",
        "historyLocations": "locations",
        "historyWorkflows": "workflows",
    }
    source_id = table_to_source.get(str(target_table))
    return load_source_configs(source_id) if source_id else fallback_bundle
