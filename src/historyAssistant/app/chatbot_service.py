from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)
from typing import Any, Generator, List, Optional, Tuple

import pandas as pd

from historyAssistant.config import YamlDict, load_source_configs, source_id_for_table
from historyAssistant.planning import REQUEST_TYPE_CHART, normalize_question
from historyAssistant.agents.execution_agent import ExecutionAgent
from historyAssistant.execution import execute_canonical_plan, open_es_connection
from historyAssistant.planning.core.validator import build_validation_policy, validate_and_repair_plan
from historyAssistant.planning.intent_planner.raw_detail_transform import build_raw_detail_plan
from historyAssistant.app.runtime import load_runtime
from historyAssistant.app.pipeline_steps import classify_and_prepare_source,finalise_pipeline_output,run_planning_stage
from historyAssistant.app.export import build_export_execution_items, prepare_first_execution_result, preview_dataframe
from historyAssistant.app.followup_resolver import apply_same_task_refinement,effective_message_from_resolution,is_confident_followup_resolution,resolve_followup
from historyAssistant.app.narration import narrate, narrate_stream, raw_record_summary_answer, table_first_answer
from historyAssistant.app.export import build_export_file
from historyAssistant.app.clarification import advance_pending_attempt,build_clarification_question,carry_followup_attempt,execute_clarification_choice,extract_choice,graceful_fallback_response,is_clarification_reply,merge_question,PendingClarification,set_clarification_debug

from historyAssistant.app.conversation_state import normalize_conversation_state

ChatHistory = List[YamlDict]


def _execute_raw_detail_followup(
    *,
    question: str,
    state: Optional[YamlDict],
    client: Any,
    export_mode: str,
    followup_llm_used: bool = True,
) -> tuple[str, List[YamlDict], pd.DataFrame, Optional[str], YamlDict] | None:
    normalized_state = normalize_conversation_state(state)
    prior_plan = normalized_state.get("last_valid_plan") or {}
    if not prior_plan:
        return None

    runtime_source_id = source_id_for_table((prior_plan.get("target_tables") or [None])[0])
    if not runtime_source_id:
        return None
    bundle = load_source_configs(runtime_source_id)
    raw_plan = build_raw_detail_plan(prior_plan, bundle=bundle)
    if not raw_plan:
        return None

    validation = validate_and_repair_plan(raw_plan, policy=build_validation_policy(bundle))
    if not validation.get("valid"):
        debug_payload = {
            "question": question,
            "followup_raw_detail": {
                "requested": True,
                "valid": False,
                "validation": validation,
                "raw_plan": raw_plan,
            },
        }
        return "The detail view could not be built from the previous result.", [], pd.DataFrame(), None, debug_payload

    canonical_plan = validation.get("canonical_plan") or raw_plan
    es = open_es_connection(config_bundle=bundle)
    execution_result = execute_canonical_plan(canonical_plan, es, config_bundle=bundle)
    execution_item: YamlDict = {
        "sub_request_id": 1,
        "sub_question": question,
        "request_type": "qa",
        "execution_result": execution_result,
        "narration": "",
    }
    prepare_first_execution_result([execution_item], es=es, bundle=bundle)
    narration_debug: YamlDict = {}
    narration = narrate(question, execution_item["execution_result"], client, force_deterministic=True, debug_sink=narration_debug)
    execution_item["narration"] = narration
    execution_item["narration_debug"] = narration_debug
    execution_items = [execution_item]
    preview_df = preview_dataframe(execution_items)
    export_items, effective_export_mode = build_export_execution_items(
        execution_items,
        export_mode=export_mode,
        es=es,
        bundle=bundle,
    )
    export_file = build_export_file(export_items, effective_export_mode)
    debug_payload = {
        "question": question,
        "planning_path": "raw_detail_followup",
        "planning": [{
            "sub_request_id": 1,
            "sub_question": question,
            "request_type": "qa",
            "canonical_plan": canonical_plan,
            "validation": validation,
            "deterministic": True,
            "fast_path": True,
            "fast_path_reason": "raw_detail_followup",
        }],
        "execution": execution_items,
        "followup_raw_detail": {
            "requested": True,
            "valid": True,
            "source_id": runtime_source_id,
            "from_plan": prior_plan,
            "canonical_plan": canonical_plan,
        },
        "llm_calls": {
            "followup": bool(followup_llm_used),
            "source_router": False,
            "intent_packet": False,
            "candidate_intent_packets": 0,
            "field_selector": False,
            "narration": False,
            "total": 1 if followup_llm_used else 0,
        },
    }
    return narration, execution_items, preview_df, export_file, debug_payload


def handle_show_source_records(
    *,
    history: Optional[ChatHistory],
    conversation_state: Optional[YamlDict],
    export_mode: str = "No export",
    show_debug: bool = True,
) -> Tuple[ChatHistory, ChatHistory, Optional[YamlDict], pd.DataFrame, Optional[str], Optional[YamlDict]]:
    chat_history: ChatHistory = list(history or [])
    question = "Show source records for the previous result."
    raw_detail = _execute_raw_detail_followup(
        question=question,
        state=conversation_state,
        client=None,
        export_mode=export_mode,
        followup_llm_used=False,
    )
    if raw_detail is None:
        answer = "No previous result is available to expand into source records."
        debug_payload: YamlDict = {
            "question": question,
            "planning_path": "source_records_button",
            "source_records_button": {"requested": True, "available": False},
            "llm_calls": {
                "followup": False,
                "source_router": False,
                "intent_packet": False,
                "candidate_intent_packets": 0,
                "field_selector": False,
                "narration": False,
                "total": 0,
            },
        }
        chat_history.append({"role": "user", "content": "Show source records"})
        chat_history.append({"role": "assistant", "content": answer})
        return chat_history, chat_history, (debug_payload if show_debug else None), pd.DataFrame(), None, None

    answer, _execution_items, preview_df, export_file, debug_payload = raw_detail
    debug_payload["source_records_button"] = {"requested": True, "available": True}
    chat_history.append({"role": "user", "content": "Show source records"})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history, chat_history, (debug_payload if show_debug else None), preview_df, export_file, None


def rewrite_last_user_message(chat_history: ChatHistory, user_message: str) -> None:
    if len(chat_history) >= 2 and chat_history[-2].get("role") == "user":
        chat_history[-2]["content"] = user_message


def handle_chat_turn(
    message: str,
    history: Optional[ChatHistory],
    date_start: str,
    date_end: str,
    time_start: str = "",
    time_end: str = "",
    export_mode: str = "No export",
    show_debug: bool = True,
    pending_clarification: Optional[YamlDict] = None,
    _skip_preplan_clarification: bool = False,
    conversation_state: Optional[YamlDict] = None,
):
    clean_message = (message or "").strip()
    original_message = clean_message
    chat_history: ChatHistory = list(history or [])
    if not clean_message:
        return chat_history, chat_history, None, pd.DataFrame(), None, pending_clarification

    pending_obj = PendingClarification.from_dict(pending_clarification)
    is_followup = bool(pending_obj and is_clarification_reply(clean_message, pending_obj))
    choice = extract_choice(clean_message, pending_obj) if is_followup and pending_obj else None
    pending = pending_obj if choice else None
    if is_followup and choice and pending:
        return handle_clarification_choice_sync(choice, pending, clean_message, chat_history, export_mode, show_debug)

    if is_followup and pending_obj and pending_obj.clarification_type != "ranking_ambiguity":
        if pending_obj.attempt_count >= pending_obj.max_attempts:
            debug_payload: YamlDict = {"question": clean_message}
            set_clarification_debug(debug_payload, pending=pending_obj, followup_answer=clean_message, reason="max_attempts_reached")
            chat_history.append({"role": "user", "content": clean_message})
            chat_history.append({"role": "assistant", "content": graceful_fallback_response()})
            return chat_history, chat_history, (debug_payload if show_debug else None), pd.DataFrame(), None, None

        bundle, _client = load_runtime()
        merged_message = merge_question(advance_pending_attempt(pending_obj), normalize_question(clean_message, bundle.vocabulary).normalized_question)
        result = handle_chat_turn(merged_message, chat_history, date_start, date_end, time_start, time_end, export_mode, show_debug, None, _skip_preplan_clarification=True)
        new_history, public_history, debug_out, preview_df, export_file, updated_pending = result
        rewrite_last_user_message(new_history, clean_message)
        if public_history is not new_history:
            rewrite_last_user_message(public_history, clean_message)
        updated_pending = carry_followup_attempt(updated_pending, pending_obj)
        if debug_out is None:
            debug_out = {"question": clean_message}
        set_clarification_debug(debug_out, pending=PendingClarification.from_dict(updated_pending) or advance_pending_attempt(pending_obj), followup_answer=clean_message, merged_question=merged_message)
        return new_history, public_history, debug_out, preview_df, export_file, updated_pending

    pending_clarification = None
    followup_resolution: Optional[YamlDict] = None
    if not _skip_preplan_clarification:
        try:
            _bundle, resolver_client = load_runtime()
            followup_resolution = resolve_followup(current_message=clean_message, history=chat_history, conversation_state=conversation_state, client=resolver_client)
            clean_message = effective_message_from_resolution(clean_message, followup_resolution)
        except Exception as exc:
            followup_resolution = {"mode": "new_question", "operation": "new_task", "confidence": 1.0, "rewrite": clean_message, "plan_patch": {}, "reason": f"followup resolver failed: {exc}"}

    bundle, client = load_runtime()
    debug_payload: YamlDict = {
        "question": clean_message,
        "controls": {"date_start": date_start, "date_end": date_end, "time_start": time_start, "time_end": time_end, "export_mode": export_mode},
    }
    if followup_resolution:
        debug_payload["followup_resolution"] = followup_resolution
    set_clarification_debug(debug_payload)

    if (
        followup_resolution
        and followup_resolution.get("operation") == "raw_detail"
        and is_confident_followup_resolution(followup_resolution)
    ):
        raw_detail = _execute_raw_detail_followup(
            question=clean_message,
            state=conversation_state,
            client=client,
            export_mode=export_mode,
        )
        if raw_detail is not None:
            answer, _execution_items, preview_df, export_file, raw_debug = raw_detail
            raw_debug["followup_resolution"] = followup_resolution
            chat_history.append({"role": "user", "content": original_message})
            chat_history.append({"role": "assistant", "content": answer})
            return chat_history, chat_history, (raw_debug if show_debug else None), preview_df, export_file, pending_clarification

    _t0 = time.monotonic()
    _source_state = conversation_state if (followup_resolution or {}).get("mode") == "followup" else None
    source_context = classify_and_prepare_source(clean_message=clean_message, bundle=bundle, client=client, debug_payload=debug_payload, conversation_state=_source_state)
    _t1 = time.monotonic()

    if source_context.classification.blocked:
        chat_history.append({"role": "user", "content": clean_message})
        chat_history.append({"role": "assistant", "content": source_context.classification.blocked_response})
        return chat_history, chat_history, (debug_payload if show_debug else None), pd.DataFrame(), None, pending_clarification

    _t2 = time.monotonic()
    planning_result, _ = run_planning_stage(clean_message=clean_message, source_context=source_context, client=client, chat_history=chat_history, date_start=date_start, date_end=date_end, time_start=time_start, time_end=time_end, skip_preplan_clarification=_skip_preplan_clarification or is_confident_followup_resolution(followup_resolution), debug_payload=debug_payload)
    _t3 = time.monotonic()

    if planning_result.clarification_needed:
        clarification_pending = planning_result.clarification_needed
        set_clarification_debug(debug_payload, pending=clarification_pending)
        chat_history.append({"role": "user", "content": clean_message})
        chat_history.append({"role": "assistant", "content": build_clarification_question(clarification_pending)})
        return chat_history, chat_history, (debug_payload if show_debug else None), pd.DataFrame(), None, clarification_pending.to_dict()

    _t4 = time.monotonic()
    exec_result = ExecutionAgent(validated_items=planning_result.validated_items, bundle=source_context.source_bundle).run()
    _t5 = time.monotonic()
    execution_items = exec_result.execution_items
    es = exec_result.es

    _t6 = time.monotonic()
    if execution_items:
        prepare_first_execution_result(execution_items, es=es, bundle=source_context.source_bundle)
        narration_debug: YamlDict = {}
        execution_items[0]["narration"] = narrate(execution_items[0]["sub_question"], execution_items[0]["execution_result"], client, force_deterministic=False, debug_sink=narration_debug)
        execution_items[0]["narration_debug"] = narration_debug
    _t7 = time.monotonic()

    _cls_steps = {s["step"]: s["duration_ms"] for s in (debug_payload.get("classification_timing") or {}).get("steps", [])}
    _, export_file, preview_df, confidence_note = finalise_pipeline_output(
        debug_payload=debug_payload, planning_result=planning_result, execution_items=execution_items,
        validated_items=planning_result.validated_items, source_context=source_context,
        timing_dict={"classification_ms": round((_t1 - _t0) * 1000), "source_routing_ms": _cls_steps.get("source_routing"), "planning_ms": round((_t3 - _t2) * 1000), "execution_ms": round((_t5 - _t4) * 1000), "narration_ms": round((_t7 - _t6) * 1000), "total_ms": round((_t7 - _t0) * 1000)},
        export_mode=export_mode, es=es,
    )

    answer_parts: List[str] = [item["narration"] for item in execution_items if item.get("narration")]
    answer_parts.extend(planning_result.blocked_answers)
    answer = "\n\n".join(answer_parts) if answer_parts else "I could classify the request, but no valid executable plan was produced."
    answer += confidence_note

    chat_history.append({"role": "user", "content": original_message})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history, chat_history, (debug_payload if show_debug else None), preview_df, export_file, pending_clarification


def handle_chat_turn_stream(
    message: str,
    history: Optional[ChatHistory],
    date_start: str,
    date_end: str,
    time_start: str,
    time_end: str,
    export_mode: str,
    show_debug: bool,
    pending_clarification: Optional[YamlDict] = None,
    _skip_preplan_clarification: bool = False,
    conversation_state: Optional[YamlDict] = None,
) -> Generator[Tuple[ChatHistory, ChatHistory, Optional[YamlDict], pd.DataFrame, Optional[str], Optional[YamlDict]], None, None]:
    clean_message = (message or "").strip()
    original_message = clean_message
    chat_history: ChatHistory = list(history or [])
    if not clean_message:
        yield chat_history, chat_history, None, pd.DataFrame(), None, pending_clarification
        return

    pending_obj = PendingClarification.from_dict(pending_clarification)
    is_followup = bool(pending_obj and is_clarification_reply(clean_message, pending_obj))
    choice = extract_choice(clean_message, pending_obj) if is_followup and pending_obj else None
    if is_followup and choice and pending_obj:
        yield from handle_clarification_choice_stream(choice, pending_obj, clean_message, chat_history, export_mode, show_debug)
        return

    if is_followup and pending_obj and pending_obj.clarification_type != "ranking_ambiguity":
        if pending_obj.attempt_count >= pending_obj.max_attempts:
            debug_payload: YamlDict = {"question": clean_message}
            set_clarification_debug(debug_payload, pending=pending_obj, followup_answer=clean_message, reason="max_attempts_reached")
            chat_history.append({"role": "user", "content": clean_message})
            chat_history.append({"role": "assistant", "content": graceful_fallback_response()})
            yield chat_history, chat_history, (debug_payload if show_debug else None), pd.DataFrame(), None, None
            return

        bundle, _client = load_runtime()
        merged_message = merge_question(advance_pending_attempt(pending_obj), normalize_question(clean_message, bundle.vocabulary).normalized_question)
        for chat, hist, debug, table, dl_file, updated_pending in handle_chat_turn_stream(merged_message, chat_history, date_start, date_end, time_start, time_end, export_mode, show_debug, None, _skip_preplan_clarification=True):
            rewrite_last_user_message(chat, clean_message)
            if hist is not chat:
                rewrite_last_user_message(hist, clean_message)
            updated_pending = carry_followup_attempt(updated_pending, pending_obj)
            if debug is None:
                debug = {"question": clean_message}
            set_clarification_debug(debug, pending=PendingClarification.from_dict(updated_pending) or advance_pending_attempt(pending_obj), followup_answer=clean_message, merged_question=merged_message)
            yield chat, hist, debug, table, dl_file, updated_pending
        return

    pending_clarification = None
    followup_resolution: Optional[YamlDict] = None
    if not _skip_preplan_clarification:
        try:
            _bundle, resolver_client = load_runtime()
            followup_resolution = resolve_followup(current_message=clean_message, history=chat_history, conversation_state=conversation_state, client=resolver_client)
            clean_message = effective_message_from_resolution(clean_message, followup_resolution)
        except Exception as exc:
            followup_resolution = {"mode": "new_question", "operation": "new_task", "confidence": 1.0, "rewrite": clean_message, "plan_patch": {}, "reason": f"followup resolver failed: {exc}"}

    t0 = time.monotonic()

    def elapsed() -> str:
        return f"{time.monotonic() - t0:.1f}s"

    chat_history.append({"role": "user", "content": original_message})
    chat_history.append({"role": "assistant", "content": f"⏳ Analyzing your question... ({elapsed()})"})
    yield chat_history, chat_history, None, pd.DataFrame(), None, pending_clarification

    try:
        yield from _stream_inner(clean_message, chat_history, date_start, date_end, time_start, time_end, export_mode, show_debug, t0, pending_clarification, _skip_preplan_clarification, followup_resolution, conversation_state)
    except Exception as exc:
        chat_history[-1]["content"] = f"An error occurred while processing your request: {exc}"
        yield chat_history, chat_history, None, pd.DataFrame(), None, pending_clarification


def _stream_inner(
    clean_message: str,
    chat_history: ChatHistory,
    date_start: str,
    date_end: str,
    time_start: str,
    time_end: str,
    export_mode: str,
    show_debug: bool,
    t0: float,
    pending_clarification: Optional[YamlDict],
    _skip_preplan_clarification: bool = False,
    followup_resolution: Optional[YamlDict] = None,
    conversation_state: Optional[YamlDict] = None,
) -> Generator[Tuple[ChatHistory, ChatHistory, Optional[YamlDict], pd.DataFrame, Optional[str], Optional[YamlDict]], None, None]:

    def elapsed() -> str:
        return f"{time.monotonic() - t0:.1f}s"

    def _yield(content: Optional[str] = None, *, preview: pd.DataFrame = pd.DataFrame(), export: Optional[str] = None, debug: Optional[YamlDict] = None, pending: Optional[YamlDict] = None) -> Tuple:
        if content is not None:
            chat_history[-1]["content"] = content
        return chat_history, chat_history, (debug if show_debug else None), preview, export, pending or pending_clarification

    bundle, client = load_runtime()
    debug_payload: YamlDict = {
        "question": clean_message,
        "controls": {"date_start": date_start, "date_end": date_end, "time_start": time_start, "time_end": time_end, "export_mode": export_mode},
    }
    if followup_resolution:
        debug_payload["followup_resolution"] = followup_resolution

    if (
        followup_resolution
        and followup_resolution.get("operation") == "raw_detail"
        and is_confident_followup_resolution(followup_resolution)
    ):
        raw_detail = _execute_raw_detail_followup(
            question=clean_message,
            state=conversation_state,
            client=client,
            export_mode=export_mode,
        )
        if raw_detail is not None:
            answer, _execution_items, preview_df, export_file, raw_debug = raw_detail
            raw_debug["followup_resolution"] = followup_resolution
            chat_history[-1]["content"] = answer + f"\n\n---\n*Completed in {elapsed()}*"
            yield chat_history, chat_history, (raw_debug if show_debug else None), preview_df, export_file, pending_clarification
            return

    if (followup_resolution and followup_resolution.get("operation") == "chart_restyle"
            and followup_resolution.get("mode") == "followup"
            and float(followup_resolution.get("confidence", 0.0)) >= 0.75):
        state = normalize_conversation_state(conversation_state)
        cached_result = (state.get("last_execution_summary") or {}).get("last_execution_result")
        new_chart_type = followup_resolution.get("plan_patch", {}).get("chart_spec", {}).get("chart_type")
        if cached_result and new_chart_type:
            from copy import deepcopy
            patched = deepcopy(cached_result)
            chart_payload = patched.get("chart_payload")
            if chart_payload and isinstance(chart_payload.get("chart_spec"), dict):
                chart_payload["chart_spec"]["chart_type"] = new_chart_type
                patched["request_type"] = REQUEST_TYPE_CHART
                execution_items = [{"sub_question": clean_message, "execution_result": patched, "narration": "chart updated"}]
                debug_payload["execution"] = execution_items
                debug_payload["chart_restyle_result"] = {"plan": state.get("last_valid_plan"), "execution_result": patched}
                yield _yield("chart updated.", preview=preview_dataframe(execution_items), debug=debug_payload)
                return

    if (followup_resolution and followup_resolution.get("operation") == "same_task_refinement"
            and followup_resolution.get("mode") == "followup"
            and float(followup_resolution.get("confidence", 0.0)) >= 0.75):
        state = normalize_conversation_state(conversation_state)
        cached_result = (state.get("last_execution_summary") or {}).get("last_execution_result")
        plan_patch = followup_resolution.get("plan_patch") or {}
        if cached_result and plan_patch:
            patched = apply_same_task_refinement(cached_result, plan_patch)
            if patched is not None:
                execution_items = [{"sub_question": clean_message, "execution_result": patched, "narration": ""}]
                narration_buffer = ""
                last_yield_len = 0
                table_first = table_first_answer(patched, clean_message)
                if table_first is not None:
                    narration_buffer = table_first
                else:
                    for chunk in narrate_stream(clean_message, patched, client, force_deterministic=False):
                        narration_buffer += chunk
                        if len(narration_buffer) - last_yield_len >= 80:
                            yield _yield(narration_buffer, debug=debug_payload)
                            last_yield_len = len(narration_buffer)
                execution_items[0]["narration"] = narration_buffer
                debug_payload["execution"] = execution_items
                debug_payload["same_task_refinement_result"] = {"plan": state.get("last_valid_plan"), "execution_result": patched}
                yield _yield(narration_buffer, preview=preview_dataframe(execution_items), debug=debug_payload)
                return

    yield _yield(f"⏳ Classifying... ({elapsed()})", debug=debug_payload)

    _t_classify = time.monotonic()
    _source_state = conversation_state if (followup_resolution or {}).get("mode") == "followup" else None
    source_context = classify_and_prepare_source(clean_message=clean_message, bundle=bundle, client=client, debug_payload=debug_payload, conversation_state=_source_state)
    _t_classify_end = time.monotonic()

    if source_context.classification.blocked:
        yield _yield(source_context.classification.blocked_response, debug=debug_payload)
        return

    yield _yield(f"⏳ Planning... ({elapsed()})", debug=debug_payload)

    _t_plan = time.monotonic()
    planning_result, _ = run_planning_stage(clean_message=clean_message, source_context=source_context, client=client, chat_history=chat_history[:-2] if len(chat_history) >= 2 else [], date_start=date_start, date_end=date_end, time_start=time_start, time_end=time_end, skip_preplan_clarification=_skip_preplan_clarification or is_confident_followup_resolution(followup_resolution), debug_payload=debug_payload)
    _t_plan_end = time.monotonic()

    if planning_result.clarification_needed:
        clarification_pending = planning_result.clarification_needed
        set_clarification_debug(debug_payload, pending=clarification_pending)
        yield _yield(build_clarification_question(clarification_pending), debug=debug_payload, pending=clarification_pending.to_dict())
        return

    yield _yield(f"⏳ Executing query... ({elapsed()})", debug=debug_payload)

    _t_exec = time.monotonic()
    exec_result = ExecutionAgent(validated_items=planning_result.validated_items, bundle=source_context.source_bundle).run()
    _t_exec_end = time.monotonic()
    execution_items = exec_result.execution_items
    es = exec_result.es

    if not execution_items:
        answer = "\n\n".join(planning_result.blocked_answers) if planning_result.blocked_answers else "I could classify the request, but no valid executable plan was produced."
        yield _yield(answer, debug=debug_payload)
        return

    narration_buffer = ""
    last_yield_len = 0

    _t_narrate = time.monotonic()
    prepare_first_execution_result(execution_items, es=es, bundle=source_context.source_bundle)
    narration_debug: YamlDict = {}
    for chunk in narrate_stream(execution_items[0]["sub_question"], execution_items[0]["execution_result"], client, force_deterministic=False, debug_sink=narration_debug):
        narration_buffer += chunk
        if len(narration_buffer) - last_yield_len >= 80:
            yield _yield(narration_buffer, debug=debug_payload)
            last_yield_len = len(narration_buffer)

    for ei in execution_items:
        ei["narration"] = narration_buffer
        ei["narration_debug"] = narration_debug
    _t_narrate_end = time.monotonic()

    _cls_steps = {s["step"]: s["duration_ms"] for s in (debug_payload.get("classification_timing") or {}).get("steps", [])}
    _, export_file, preview_df, confidence_note = finalise_pipeline_output(
        debug_payload=debug_payload, planning_result=planning_result, execution_items=execution_items,
        validated_items=planning_result.validated_items, source_context=source_context,
        timing_dict={"classification_ms": round((_t_classify_end - _t_classify) * 1000), "source_routing_ms": _cls_steps.get("source_routing"), "planning_ms": round((_t_plan_end - _t_plan) * 1000), "execution_ms": round((_t_exec_end - _t_exec) * 1000), "narration_ms": round((_t_narrate_end - _t_narrate) * 1000), "total_ms": round((_t_narrate_end - _t_classify) * 1000)},
        export_mode=export_mode, es=es,
    )

    if planning_result.blocked_answers:
        narration_buffer = (narration_buffer + "\n\n" + "\n\n".join(planning_result.blocked_answers)).strip()
    narration_buffer += confidence_note
    chat_history[-1]["content"] = narration_buffer + f"\n\n---\n*Completed in {elapsed()}*"
    yield chat_history, chat_history, (debug_payload if show_debug else None), preview_df, export_file, pending_clarification


def handle_clarification_choice_sync(
    choice: str,
    pending: PendingClarification,
    clean_message: str,
    chat_history: ChatHistory,
    export_mode: str,
    show_debug: bool,
) -> Tuple[ChatHistory, ChatHistory, Optional[YamlDict], pd.DataFrame, Optional[str], Optional[YamlDict]]:
    bundle, client = load_runtime()
    answer, preview_df, export_file, debug_payload = execute_clarification_choice(choice, pending, bundle=bundle, client=client, export_mode=export_mode)
    set_clarification_debug(debug_payload, pending=pending, followup_answer=clean_message)
    chat_history.append({"role": "user", "content": clean_message})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history, chat_history, (debug_payload if show_debug else None), preview_df, export_file, None


def handle_clarification_choice_stream(
    choice: str,
    pending: PendingClarification,
    clean_message: str,
    chat_history: ChatHistory,
    export_mode: str,
    show_debug: bool,
) -> Generator[Tuple[ChatHistory, ChatHistory, Optional[YamlDict], pd.DataFrame, Optional[str], Optional[YamlDict]], None, None]:
    t0 = time.monotonic()

    def elapsed() -> str:
        return f"{time.monotonic() - t0:.1f}s"

    bundle, client = load_runtime()
    chat_history.append({"role": "user", "content": clean_message})
    chat_history.append({"role": "assistant", "content": f"⏳ Applying your choice... ({elapsed()})"})
    yield chat_history, chat_history, None, pd.DataFrame(), None, None

    answer, preview_df, export_file, debug_payload = execute_clarification_choice(choice, pending, bundle=bundle, client=client, export_mode=export_mode)
    set_clarification_debug(debug_payload, pending=pending, followup_answer=clean_message)
    chat_history[-1]["content"] = answer + f"\n\n---\n*Completed in {elapsed()}*"
    yield chat_history, chat_history, (debug_payload if show_debug else None), preview_df, export_file, None
