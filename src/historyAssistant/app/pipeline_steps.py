from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

from historyAssistant.config import YamlDict, load_source_configs, source_id_for_table
from historyAssistant.execution.executor import get_es_index
from historyAssistant.planning import build_validation_policy, normalize_question, validate_and_repair_plan
from historyAssistant.agents.classification_agent import ClassificationAgent
from historyAssistant.planning.intent_planner.slot_planner import try_build_slot_plan
from historyAssistant.planning.intent_planner.candidate_arbitration import arbitrate_low_confidence_source
from historyAssistant.app.source_plan_repairs import (
    active_date_field,
    active_timestamp_field,
    repair_location_person_lookup,
    repair_raw_records_sort,
    repair_workflow_presence_lookup,
    should_apply_picker_date_fallback,
)
from historyAssistant.app.filters import apply_date_fallback, apply_time_fallback
from historyAssistant.app.clarification import PendingClarification, record_deferred_ranking_clarification
from historyAssistant.app.conversation_state import normalize_conversation_state

ChatHistory = List[YamlDict]


@dataclass
class SourceContext:
    classification: Any
    source_id: str
    source_bundle: Any
    normalized_message: str
    detected_entities: YamlDict
    request_type: str
    preferred_chart_type: Optional[str]
    chart_suppressed: bool
    chart_intent_source: Optional[str]
    canonical_table: Optional[str]


@dataclass
class PlanningResult:
    clarification_needed: Optional[PendingClarification] = None
    validated_items: List[YamlDict] = field(default_factory=list)
    blocked_answers: List[str] = field(default_factory=list)
    planning_outputs: List[YamlDict] = field(default_factory=list)
    planning_hints: Any = None
    debug_updates: YamlDict = field(default_factory=dict)


def classify_and_prepare_source(
    *,
    clean_message: str,
    bundle: Any,
    client: Any,
    debug_payload: YamlDict,
    conversation_state: Optional[YamlDict] = None,
) -> SourceContext:
    state = normalize_conversation_state(conversation_state)
    cached_source_id: Optional[str] = None
    if state.get("last_source_id") and float(state.get("last_source_confidence") or 0.0) >= 0.85:
        cached_source_id = state["last_source_id"]

    classification = ClassificationAgent(
        clean_message=clean_message,
        bundle=bundle,
        client=client,
        cached_source_id=cached_source_id,
    ).run()
    debug_payload.update(classification.debug_updates)

    source_id = classification.source_ids[0] if classification.source_ids else "calls"
    source_bundle = load_source_configs(source_id)
    source_normalization = normalize_question(clean_message, source_bundle.vocabulary)
    canonical_table = classification.canonical_tables[0] if classification.canonical_tables else None

    debug_payload["selected_source"] = {
        "source_id": source_id,
        "source_ids": list(classification.source_ids or []),
        "canonical_table": canonical_table,
        "canonical_tables": list(classification.canonical_tables or []),
        "es_index": get_es_index(config_bundle=source_bundle, canonical_table=canonical_table),
        "source_confidence": classification.source_confidence,
    }
    debug_payload["source_normalization"] = {
        "normalized": source_normalization.normalized_question,
        "replacements": [
            {"matched": r[0], "canonical": r[1], "category": r[2]}
            for r in source_normalization.replacements
        ],
        "detected_entities": source_normalization.detected_entities,
    }

    return SourceContext(
        classification=classification,
        source_id=source_id,
        source_bundle=source_bundle,
        normalized_message=source_normalization.normalized_question,
        detected_entities=source_normalization.detected_entities,
        request_type=classification.request_type,
        preferred_chart_type=classification.preferred_chart_type,
        chart_suppressed=classification.chart_suppressed,
        chart_intent_source=classification.chart_intent_source,
        canonical_table=canonical_table,
    )


def _finalize_plan(
    plan: YamlDict,
    *,
    clean_message: str,
    detected_entities: YamlDict,
    fallback_bundle: Any,
    date_start: str,
    date_end: str,
    time_start: str,
    time_end: str,
) -> tuple[YamlDict, Any]:
    """Apply source-specific repairs and picker date/time fallbacks to a slot plan.

    Shared by the single-source path and the source-ambiguity candidate plans so
    both reach the executor in the same execution-ready state. Without this,
    candidate plans skip the picker date fallback and reach Elasticsearch with an
    unresolved date range.
    """
    plan = dict(plan)
    runtime_source_id = source_id_for_table((plan.get("target_tables") or [None])[0])
    bundle = load_source_configs(runtime_source_id) if runtime_source_id else fallback_bundle
    if runtime_source_id == "locations":
        plan = repair_location_person_lookup(plan, question=clean_message, bundle=bundle)
    elif runtime_source_id == "workflows":
        plan = repair_workflow_presence_lookup(plan, question=clean_message, bundle=bundle)

    plan = repair_raw_records_sort(
        plan,
        question=clean_message,
        bundle=bundle,
        detected_entities=detected_entities,
    )

    if should_apply_picker_date_fallback(plan):
        plan = apply_date_fallback(
            plan,
            picker_start=date_start,
            picker_end=date_end,
            date_field=active_date_field(bundle),
        )
    plan = apply_time_fallback(
        plan,
        picker_time_start=time_start,
        picker_time_end=time_end,
        timestamp_field=active_timestamp_field(bundle),
    )
    return plan, bundle


def run_planning_stage(
    *,
    clean_message: str,
    source_context: SourceContext,
    client: Any,
    chat_history: ChatHistory,
    date_start: str,
    date_end: str,
    time_start: str,
    time_end: str,
    skip_preplan_clarification: bool,
    debug_payload: YamlDict,
) -> tuple[PlanningResult, bool]:
    planning_result = PlanningResult()
    slot_result = try_build_slot_plan(
        question=clean_message,
        normalized_question=source_context.normalized_message,
        source_id=source_context.source_id,
        target_table=source_context.canonical_table or "",
        detected_entities=source_context.detected_entities,
        request_type=source_context.request_type,
        chart_suppressed=source_context.chart_suppressed,
        preferred_chart_type=source_context.preferred_chart_type,
        bundle=source_context.source_bundle,
        client=client,
        use_llm_slots=True,
    )
    arbitration = arbitrate_low_confidence_source(
        question=clean_message,
        selected_source_id=source_context.source_id,
        selected_result=slot_result,
        source_confidence=float(source_context.classification.source_confidence or 0.0),
        request_type=source_context.request_type,
        chart_suppressed=source_context.chart_suppressed,
        preferred_chart_type=source_context.preferred_chart_type,
        client=client,
    )
    if arbitration.replacement is not None:
        slot_result = arbitration.replacement.result
    slot_debug = slot_result.to_debug_dict()
    if arbitration.reason not in {"source_confident", "client_missing"} or arbitration.candidates:
        slot_debug["source_arbitration"] = arbitration.to_debug_dict()
    debug_payload["slot_planner"] = slot_debug
    planning_result.debug_updates = {
        "slot_planner": slot_debug,
        "planning_path": "intent_planner",
        "answer_shape": slot_debug.get("answer_shape"),
        "slot_reason": slot_result.reason,
        "slot_used": bool(slot_result.used),
    }

    if arbitration.should_clarify:
        # Run each candidate through the same repair + picker-fallback pipeline as
        # the main path, so whichever one the user picks is execution-ready.
        finalized_candidates = [
            _finalize_plan(
                candidate.canonical_plan or {},
                clean_message=clean_message,
                detected_entities=source_context.detected_entities,
                fallback_bundle=source_context.source_bundle,
                date_start=date_start,
                date_end=date_end,
                time_start=time_start,
                time_end=time_end,
            )[0]
            for candidate in arbitration.candidates
        ]
        planning_result.clarification_needed = PendingClarification(
            clarification_type="source_ambiguity",
            original_question=clean_message,
            normalized_question=source_context.normalized_message,
            candidate_interpretations=[candidate.interpretation for candidate in arbitration.candidates],
            canonical_plan=finalized_candidates[0],
            candidate_plans=finalized_candidates,
        )
        debug_payload.update(planning_result.debug_updates)
        return planning_result, False

    if not slot_result.used or not slot_result.canonical_plan:
        validation = slot_result.validation or {}
        blocking = list(validation.get("blocking_errors") or [])
        planning_result.blocked_answers.extend(blocking or [slot_result.reason or "No valid executable plan was produced."])
        debug_payload.update(planning_result.debug_updates)
        return planning_result, False

    plan, bundle = _finalize_plan(
        slot_result.canonical_plan,
        clean_message=clean_message,
        detected_entities=source_context.detected_entities,
        fallback_bundle=source_context.source_bundle,
        date_start=date_start,
        date_end=date_end,
        time_start=time_start,
        time_end=time_end,
    )

    validation = validate_and_repair_plan(plan, policy=build_validation_policy(bundle))
    if not validation.get("valid"):
        planning_result.blocked_answers.extend(
            list(validation.get("blocking_errors") or []) or ["The generated plan did not pass validation."]
        )
        debug_payload.update(planning_result.debug_updates)
        return planning_result, False

    plan = validation.get("canonical_plan") or plan
    validation["canonical_plan"] = plan
    item = {
        "sub_request_id": 1,
        "text": clean_message,
        "display_text": clean_message,
        "request_type": source_context.request_type,
        "preferred_chart_type": source_context.preferred_chart_type,
        "chart_suppressed": source_context.chart_suppressed,
        "chart_intent_source": source_context.chart_intent_source,
    }
    planning_result.validated_items.append({"item": item, "validation": validation, "canonical_plan": plan})
    planning_result.planning_outputs.append({
        "sub_request_id": 1,
        "sub_question": clean_message,
        "request_type": source_context.request_type,
        "canonical_plan": plan,
        "validation": validation,
        "deterministic": slot_result.reason != "llm_slots",
        "fast_path": slot_result.reason != "llm_slots",
        "fast_path_reason": slot_result.reason,
    })

    debug_payload.update(planning_result.debug_updates)
    return planning_result, True


def summarize_llm_calls(debug_payload: YamlDict, execution_items: List[YamlDict]) -> YamlDict:
    """Consolidated per-turn map of which LLM calls actually fired.

    Derived from the debug evidence each stage already records, so it costs
    nothing extra and stays accurate even as the gating logic evolves. Use it
    to reason about per-turn LLM latency (plan steps 9 & 10).
    """
    # Short-circuit follow-up resolutions return without a "payload"; the LLM
    # branch attaches one on both success and failure.
    followup = debug_payload.get("followup_resolution") or {}
    followup_fired = "payload" in followup

    # router == "llm" means the LLM picked the source; "local_fallback" means
    # the LLM was called but raised. Both count as a call that happened.
    router_kind = ((debug_payload.get("source_routing") or {}).get("debug") or {}).get("router")
    source_router_fired = router_kind in {"llm", "local_fallback"}

    # The slot planner records an "llm_slot_extraction" entry only when the
    # single unified planning LLM ran.
    intent_packet_fired = "llm_slot_extraction" in (debug_payload.get("slot_planner") or {})
    arbitration = ((debug_payload.get("slot_planner") or {}).get("source_arbitration") or {})
    arbitration_candidates = list(arbitration.get("candidates") or [])
    arbitration_packets = sum(
        1
        for candidate in arbitration_candidates
        if isinstance(candidate, dict) and candidate.get("llm_slot_fired")
    )
    additional_arbitration_packets = max(0, arbitration_packets - (1 if intent_packet_fired else 0))

    narration_debug: YamlDict = {}
    if execution_items:
        narration_debug = execution_items[0].get("narration_debug") or {}
    # The field selector attaches "field_selection" only after it is invoked;
    # intent-packet selections are reused planner output and do not issue a call.
    field_sel = narration_debug.get("field_selection")
    field_selector_fired = (
        isinstance(field_sel, dict)
        and field_sel.get("reason") not in {
            "client_missing",
            "intent_packet_field_selection",
            "intent_packet_missing_fields",
            "intent_packet_fields_not_in_result",
            "no_records",
            "no_available_fields",
        }
    )
    # The LLM narration path attaches "payload_kind"; deterministic/template
    # paths do not.
    narration_fired = "payload_kind" in narration_debug

    calls = {
        "followup": bool(followup_fired),
        "source_router": bool(source_router_fired),
        "intent_packet": bool(intent_packet_fired),
        "candidate_intent_packets": additional_arbitration_packets,
        "field_selector": bool(field_selector_fired),
        "narration": bool(narration_fired),
    }
    calls["total"] = (
        sum(1 for key, fired in calls.items() if key != "candidate_intent_packets" and fired)
        + additional_arbitration_packets
    )
    return calls


def _plan_overrode_source(execution_items: List[YamlDict], source_context: SourceContext) -> bool:
    if not execution_items:
        return False
    plan = (execution_items[0].get("execution_result") or {}).get("canonical_plan") or {}
    plan_source = source_id_for_table((plan.get("target_tables") or [None])[0])
    return bool(plan_source and plan_source != source_context.source_id)


def finalise_pipeline_output(
    *,
    debug_payload: YamlDict,
    planning_result: PlanningResult,
    execution_items: List[YamlDict],
    validated_items: List[YamlDict],
    source_context: SourceContext,
    timing_dict: YamlDict,
    export_mode: str,
    es: Any,
) -> tuple[List[YamlDict], Optional[str], pd.DataFrame, str]:
    from historyAssistant.app.export import build_export_execution_items, build_export_file, preview_dataframe

    debug_payload["_timing"] = timing_dict
    debug_payload["planning"] = planning_result.planning_outputs
    debug_payload["execution"] = execution_items
    debug_payload["llm_calls"] = summarize_llm_calls(debug_payload, execution_items)

    for vi in validated_items:
        record_deferred_ranking_clarification(
            debug_payload,
            vi.get("validation", {}).get("canonical_plan", {}) or {},
        )

    confidence_note = ""
    if source_context.classification.source_confidence < 0.8 and not _plan_overrode_source(execution_items, source_context):
        confidence_note = "\n\n> **Note:** The data-source classification was uncertain, so the result may not fully match your intent."

    export_items, effective_export_mode = build_export_execution_items(
        execution_items, export_mode=export_mode, es=es, bundle=source_context.source_bundle,
    )
    export_file = build_export_file(export_items, effective_export_mode)
    preview_df = preview_dataframe(execution_items)
    return export_items, export_file, preview_df, confidence_note
