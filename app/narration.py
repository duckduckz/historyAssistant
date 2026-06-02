from __future__ import annotations

import json as _json
from typing import Any, Generator

from historyAssistant.config import YamlDict, load_all_configs
from historyAssistant.execution.result.deterministic_narration import (
    deterministic_narration,
    display_label,
    format_execution_error,
    format_value,
    try_template_ranked_single_result,
)
from historyAssistant.execution.result.narration_payloads import (
    build_comparison_payload,
    build_fast_narration_payload,
    build_llm_narration_payload,
    fallback_narration_after_llm_error,
    payload_debug,
)
from historyAssistant.execution.result.narration_field_selector import (
    NarrationFieldSelection,
    select_narration_fields_from_intent_packet,
)
from historyAssistant.execution.result.raw_record_answers import (
    needs_full_result_download,
    raw_record_summary_answer,
    table_first_answer,
)
from historyAssistant.llm import NARRATION_SYSTEM_TEXT
from historyAssistant.llm.llm_client import _json_default, call_json_prompt, call_prompt_stream, get_default_model_name


def get_narration_mode() -> str:
    try:
        bundle = load_all_configs()
        runtime_cfg = bundle.app_config.get("runtime", {})
        return str(runtime_cfg.get("narration_mode", "hybrid")).strip().lower()
    except Exception:
        return "hybrid"


def _should_use_compact_narration(execution_result: YamlDict) -> bool:
    result_type = execution_result.get("result_type", "aggregated")
    return (
        result_type in {"raw_records", "aggregated"}
        or execution_result.get("scalar_answer") is not None
    )


def should_use_deterministic_narration(execution_result: YamlDict, *, force_deterministic: bool = False) -> bool:
    return (
        bool(force_deterministic)
        or bool(execution_result.get("error"))
        or bool(execution_result.get("warnings"))
        or execution_result.get("result_type") == "raw_records"
    )


def _pick_narration_strategy(
    question: str,
    execution_result: YamlDict,
    *,
    force_deterministic: bool,
    debug_sink: YamlDict | None,
    field_selection: NarrationFieldSelection | None = None,
) -> tuple[str, Any]:
    ranked_single = try_template_ranked_single_result(execution_result)
    if ranked_single:
        if debug_sink is not None:
            debug_sink.update({"mode": "template", "reason": "ranked_single_group_result"})
        return "ready", ranked_single

    if should_use_deterministic_narration(execution_result, force_deterministic=force_deterministic):
        if debug_sink is not None:
            debug_sink.update({"mode": "deterministic", "reason": "forced_or_error_or_warning"})
        return "ready", deterministic_narration(question, execution_result, field_selection=field_selection)

    mode = get_narration_mode()
    if mode == "deterministic":
        if debug_sink is not None:
            debug_sink.update({"mode": "deterministic", "reason": "runtime_mode"})
        return "ready", deterministic_narration(question, execution_result, field_selection=field_selection)

    if mode == "llm" or not _should_use_compact_narration(execution_result):
        payload = build_llm_narration_payload(question, execution_result, field_selection=field_selection)
        if debug_sink is not None:
            debug_sink.update(payload_debug(payload, mode=mode, payload_kind="full"))
        return "full", payload

    payload = build_fast_narration_payload(question, execution_result, field_selection=field_selection)
    if debug_sink is not None:
        debug_sink.update(payload_debug(payload, mode=mode, payload_kind="fast_compact"))
    return "compact", payload


def narrate(
    question: str,
    execution_result: YamlDict,
    client: Any,
    *,
    force_deterministic: bool = False,
    debug_sink: YamlDict | None = None,
) -> str:
    field_selection = _select_fields_for_narration(
        execution_result=execution_result,
        debug_sink=debug_sink,
    )
    strategy, value = _pick_narration_strategy(
        question,
        execution_result,
        force_deterministic=force_deterministic,
        debug_sink=debug_sink,
        field_selection=field_selection,
    )
    if strategy == "ready":
        return value
    try:
        return call_json_prompt(
            client,
            model=get_default_model_name(),
            system_text=NARRATION_SYSTEM_TEXT,
            user_payload=value,
        )
    except Exception:
        if strategy == "full":
            return fallback_narration_after_llm_error(execution_result)
        return deterministic_narration(question, execution_result, field_selection=field_selection)


def narrate_stream(
    question: str,
    execution_result: YamlDict,
    client: Any,
    *,
    force_deterministic: bool = False,
    debug_sink: YamlDict | None = None,
) -> Generator[str, None, None]:
    field_selection = _select_fields_for_narration(
        execution_result=execution_result,
        debug_sink=debug_sink,
    )
    strategy, value = _pick_narration_strategy(
        question,
        execution_result,
        force_deterministic=force_deterministic,
        debug_sink=debug_sink,
        field_selection=field_selection,
    )
    if strategy == "ready":
        yield value
        return

    user_text = _json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)
    try:
        for chunk in call_prompt_stream(
            client,
            model=get_default_model_name(),
            system_text=NARRATION_SYSTEM_TEXT,
            user_text=user_text,
        ):
            yield chunk
    except Exception:
        if strategy == "full":
            yield fallback_narration_after_llm_error(execution_result)
        else:
            yield deterministic_narration(question, execution_result, field_selection=field_selection)


def _select_fields_for_narration(
    *,
    execution_result: YamlDict,
    debug_sink: YamlDict | None,
) -> NarrationFieldSelection | None:
    if execution_result.get("error") or execution_result.get("empty"):
        return None
    if execution_result.get("result_type") == "raw_records":
        return None
    if not execution_result.get("records"):
        return None
    # Field hints come for free from the plan's intent packet: LLM-planned
    # questions already carry requested/summary/display fields, and deterministic
    # plans simply return unused. We never make a separate field-selection LLM
    # call — the single narration call below produces the prose from the full
    # records, so analytic answers cost one LLM call instead of two and
    # deterministic/template answers cost none.
    intent_selection = select_narration_fields_from_intent_packet(execution_result=execution_result)
    if debug_sink is not None:
        debug_sink["field_selection"] = intent_selection.to_dict()
    return intent_selection if intent_selection.used else None


__all__ = [
    "build_comparison_payload",
    "build_fast_narration_payload",
    "build_llm_narration_payload",
    "deterministic_narration",
    "display_label",
    "fallback_narration_after_llm_error",
    "format_execution_error",
    "format_value",
    "get_narration_mode",
    "narrate",
    "narrate_stream",
    "needs_full_result_download",
    "raw_record_summary_answer",
    "should_use_deterministic_narration",
    "table_first_answer",
    "try_template_ranked_single_result",
]
