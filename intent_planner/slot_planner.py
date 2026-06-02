from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, Optional

from historyAssistant.config import ConfigBundle

from .answer_shape import ANALYTIC_SUMMARY, DISTINCT_VALUES, RAW_RECORDS, resolve_answer_shape
from .intent_extractor import extract_intent_slots
from .llm_slot_extractor import extract_llm_intent_slots
from .neutral_intent import resolve_source_intent
from .plan_builder import build_plan_from_slots
from .record_intent import extract_record_intent_slots
from .slot_quality import assess_slot_quality

YamlDict = Dict[str, Any]

_validator = import_module("..core.validator", __package__)
_build_validation_policy = _validator.build_validation_policy
_validate_and_repair_plan = _validator.validate_and_repair_plan


@dataclass(frozen=True)
class SlotPlannerResult:
    used: bool
    canonical_plan: Optional[YamlDict] = None
    validation: Optional[YamlDict] = None
    reason: str = ""
    debug: YamlDict = field(default_factory=dict)

    def to_debug_dict(self) -> YamlDict:
        return {
            "used": self.used,
            "reason": self.reason,
            "canonical_plan_preview": _preview_plan(self.canonical_plan),
            "validation_valid": bool(self.validation and self.validation.get("valid")),
            **dict(self.debug),
        }


def try_build_slot_plan(
    *,
    question: str,
    normalized_question: str,
    source_id: str,
    target_table: str,
    detected_entities: Optional[YamlDict],
    request_type: str,
    chart_suppressed: bool,
    preferred_chart_type: Optional[str],
    bundle: ConfigBundle,
    client: Any = None,
    use_llm_slots: bool = False,
) -> SlotPlannerResult:
    source_resolution = resolve_source_intent(
        question=question,
        initial_source_id=source_id,
        initial_target_table=target_table,
        initial_bundle=bundle,
        initial_normalized_question=normalized_question,
        initial_detected_entities=detected_entities,
    )
    source_id = source_resolution.source_id
    target_table = source_resolution.target_table
    bundle = source_resolution.bundle
    normalized_question = source_resolution.normalized_question
    detected_entities = source_resolution.detected_entities

    # Always resolve answer shape deterministically — the LLM slot extractor covers shape inference when it is called below, so no separate LLM call here.
    shape = resolve_answer_shape(
        question=question,
        normalized_question=normalized_question,
        detected_entities=detected_entities,
        bundle=bundle,
        client=None,
    )
    shape_debug = {
        "source_intent_resolution": source_resolution.to_dict(),
        "answer_shape": shape.to_dict(),
    }

    if shape.shape in {RAW_RECORDS, DISTINCT_VALUES}:
        # Deterministic record path — no LLM projection call.
        # If deterministic projection fails and use_llm_slots is on, we fall through to the unified LLM slot call below rather than making a separate projection call.
        record_result = _try_record_plan(
            question=question,
            normalized_question=normalized_question,
            source_id=source_id,
            target_table=target_table,
            bundle=bundle,
            detected_entities=detected_entities,
            request_type=request_type,
            chart_suppressed=chart_suppressed,
            preferred_chart_type=preferred_chart_type,
            client=client,
            requested_answer_shape=shape.shape,
            debug_prefix=shape_debug,
        )
        if record_result.used or not use_llm_slots:
            return record_result
        # Deterministic record extraction failed — fall through to unified LLM call.

    # Deterministic analytic-summary extraction (no separate LLM filter call).
    extraction = extract_intent_slots(
        question=question,
        normalized_question=normalized_question,
        source_id=source_id,
        target_table=target_table,
        detected_entities=detected_entities,
        request_type=request_type,
        chart_suppressed=chart_suppressed,
        bundle=bundle,
        client=client,
    )
    debug: YamlDict = {
        **shape_debug,
        "slot_extraction": extraction.to_debug_dict(),
    }
    quality = assess_slot_quality(
        question=question,
        normalized_question=normalized_question,
        detected_entities=detected_entities,
        source_resolution=source_resolution,
        extraction=extraction,
    )
    debug["slot_quality"] = quality.to_dict()

    if not extraction.used and (extraction.reason or "").startswith("raw_metric_not_slot_planned"):
        record_result = _try_record_plan(
            question=question,
            normalized_question=normalized_question,
            source_id=source_id,
            target_table=target_table,
            bundle=bundle,
            detected_entities=detected_entities,
            request_type=request_type,
            chart_suppressed=chart_suppressed,
            preferred_chart_type=preferred_chart_type,
            client=client,
            requested_answer_shape=RAW_RECORDS,
            debug_prefix=shape_debug,
        )
        if record_result.used or not use_llm_slots:
            return record_result
        # Fall through to unified LLM call if deterministic record path also fails.

    # A confident analytic classification (a computed number/ranking the user
    # clearly asked for) must resolve a metric or fail — never silently become a
    # record dump. Anything else may fall back to a lookup.
    confident_analytic = shape.shape == ANALYTIC_SUMMARY and shape.confidence >= 0.8

    # Deterministic lookup BEFORE the LLM: when no metric resolved and the
    # question is not a confident analytic ask, a raw-records lookup is the
    # natural answer. If the deterministic record path produces a plan with real
    # constraints (a filter, date, or time), trust it and skip the LLM entirely —
    # confident lookups then cost zero LLM calls. General and structural: gated on
    # the plan's shape, not on any keyword, regex, or config rule.
    lookup_result: Optional[SlotPlannerResult] = None
    if (not extraction.used or not extraction.slots) and not confident_analytic:
        lookup_result = _try_record_plan(
            question=question,
            normalized_question=normalized_question,
            source_id=source_id,
            target_table=target_table,
            bundle=bundle,
            detected_entities=detected_entities,
            request_type=request_type,
            chart_suppressed=chart_suppressed,
            preferred_chart_type=preferred_chart_type,
            client=client,
            requested_answer_shape=RAW_RECORDS,
            debug_prefix=debug,
        )
        if lookup_result.used and _plan_has_constraints(lookup_result.canonical_plan):
            return lookup_result

    # Single unified LLM planning call — covers answer_shape, metric, group_by, filters, dates, projection_field, and comparison in one round trip.
    if use_llm_slots and (not extraction.used or not extraction.slots or not quality.trust_deterministic):
        llm_extraction = extract_llm_intent_slots(
            question=question,
            normalized_question=normalized_question,
            source_id=source_id,
            target_table=target_table,
            request_type=request_type,
            bundle=bundle,
            client=client,
        )
        debug["llm_slot_extraction"] = llm_extraction.to_debug_dict()
        if llm_extraction.used and llm_extraction.slots:
            # If LLM determined this is a record query, build that plan directly.
            if llm_extraction.slots.answer_shape in {RAW_RECORDS, DISTINCT_VALUES}:
                plan, validation = _build_and_validate(
                    slots=llm_extraction.slots,
                    question=question,
                    bundle=bundle,
                    chart_suppressed=chart_suppressed,
                    preferred_chart_type=preferred_chart_type,
                )
                debug["validation"] = _validation_debug(validation)
                if not validation.get("valid"):
                    return SlotPlannerResult(
                        False,
                        canonical_plan=plan,
                        validation=validation,
                        reason="llm_record_plan_failed_validation",
                        debug=debug,
                    )
                return SlotPlannerResult(
                    True,
                    canonical_plan=validation.get("canonical_plan") or plan,
                    validation=validation,
                    reason=llm_extraction.reason,
                    debug=debug,
                )
            extraction = llm_extraction

    if not extraction.used or not extraction.slots:
        # Final fallback: neither deterministic nor LLM extraction resolved a
        # metric. A confident analytic ask must fail rather than become a record
        # dump; anything else falls back to the lookup. Reuse the deterministic
        # lookup computed before the LLM (it didn't short-circuit only because it
        # lacked strong constraints) so we never run the record path twice.
        if confident_analytic:
            return SlotPlannerResult(False, reason=extraction.reason, debug=debug)
        if lookup_result is None:
            lookup_result = _try_record_plan(
                question=question,
                normalized_question=normalized_question,
                source_id=source_id,
                target_table=target_table,
                bundle=bundle,
                detected_entities=detected_entities,
                request_type=request_type,
                chart_suppressed=chart_suppressed,
                preferred_chart_type=preferred_chart_type,
                client=client,
                requested_answer_shape=RAW_RECORDS,
                debug_prefix=debug,
            )
        if lookup_result.used:
            return lookup_result
        return SlotPlannerResult(False, reason=extraction.reason, debug=debug)

    plan, validation = _build_and_validate(
        slots=extraction.slots,
        question=question,
        bundle=bundle,
        chart_suppressed=chart_suppressed,
        preferred_chart_type=preferred_chart_type,
    )
    debug["validation"] = _validation_debug(validation)

    # Retry with LLM only when deterministic produced a plan that failed validation and LLM has not been called yet in this turn.
    if use_llm_slots and _should_try_llm_after_deterministic(
        question=question,
        detected_entities=detected_entities,
        validation=validation,
        debug=debug,
    ):
        llm_extraction = extract_llm_intent_slots(
            question=question,
            normalized_question=normalized_question,
            source_id=source_id,
            target_table=target_table,
            request_type=request_type,
            bundle=bundle,
            client=client,
        )
        debug["llm_slot_extraction"] = llm_extraction.to_debug_dict()
        if llm_extraction.used and llm_extraction.slots:
            llm_plan, llm_validation = _build_and_validate(
                slots=llm_extraction.slots,
                question=question,
                bundle=bundle,
                chart_suppressed=chart_suppressed,
                preferred_chart_type=preferred_chart_type,
            )
            debug["llm_validation"] = _validation_debug(llm_validation)
            if llm_validation.get("valid"):
                plan = llm_plan
                validation = llm_validation
                extraction = llm_extraction

    if not validation.get("valid"):
        return SlotPlannerResult(
            False,
            canonical_plan=plan,
            validation=validation,
            reason="slot_plan_failed_validation",
            debug=debug,
        )

    return SlotPlannerResult(
        True,
        canonical_plan=validation.get("canonical_plan") or plan,
        validation=validation,
        reason=extraction.reason,
        debug=debug,
    )


def _try_record_plan(
    *,
    question: str,
    normalized_question: str,
    source_id: str,
    target_table: str,
    bundle: ConfigBundle,
    detected_entities: Optional[YamlDict],
    request_type: str,
    chart_suppressed: bool,
    preferred_chart_type: Optional[str],
    client: Any = None,
    requested_answer_shape: Optional[str] = None,
    debug_prefix: Optional[YamlDict] = None,
) -> SlotPlannerResult:
    try:
        extraction = extract_record_intent_slots(
            question=question,
            normalized_question=normalized_question,
            source_id=source_id,
            target_table=target_table,
            detected_entities=detected_entities,
            request_type=request_type,
            bundle=bundle,
            client=client,
            requested_answer_shape=requested_answer_shape,
        )
    except Exception as exc:
        return SlotPlannerResult(False, reason=f"record_intent_failed:{exc}", debug=dict(debug_prefix or {}))

    if not extraction.used or not extraction.slots:
        return SlotPlannerResult(
            False,
            reason=extraction.reason or "record_intent_empty",
            debug={**dict(debug_prefix or {}), "record_intent": extraction.to_debug_dict()},
        )

    plan, validation = _build_and_validate(
        slots=extraction.slots,
        question=question,
        bundle=bundle,
        chart_suppressed=chart_suppressed,
        preferred_chart_type=preferred_chart_type,
    )
    debug: YamlDict = {
        **(debug_prefix or {}),
        "record_intent": extraction.to_debug_dict(),
        "record_plan": _preview_plan(plan),
        "validation": _validation_debug(validation),
    }

    if not validation.get("valid"):
        return SlotPlannerResult(
            False,
            canonical_plan=plan,
            validation=validation,
            reason="record_intent_failed_validation",
            debug=debug,
        )

    return SlotPlannerResult(
        True,
        canonical_plan=validation.get("canonical_plan") or plan,
        validation=validation,
        reason=extraction.reason or "deterministic_record_intent",
        debug=debug,
    )


def _plan_has_constraints(plan: Optional[YamlDict]) -> bool:
    """True when a raw-records plan is a targeted lookup rather than an open dump.

    Structural check (filter / date / time present) — no text or keyword matching.
    Used to decide whether a deterministic lookup is confident enough to skip the
    LLM.
    """
    if not plan:
        return False
    if plan.get("filters"):
        return True
    if plan.get("date_range") or (plan.get("date_rule") and plan.get("date_rule") != "none"):
        return True
    if plan.get("time_range"):
        return True
    return False


def _preview_plan(plan: Optional[YamlDict]) -> Optional[YamlDict]:
    if not plan:
        return None
    return {
        "request_type": plan.get("request_type"),
        "metric": plan.get("metric"),
        "aggregation": plan.get("aggregation"),
        "group_by": plan.get("group_by"),
        "filters": plan.get("filters"),
        "source_fields": plan.get("source_fields"),
        "order_by": plan.get("order_by"),
        "limit": plan.get("limit"),
        "date_rule": plan.get("date_rule"),
        "date_range": plan.get("date_range"),
        "compare_to": plan.get("compare_to"),
        "bucket_interval": plan.get("bucket_interval"),
        "bucket_summary": plan.get("bucket_summary"),
        "query_shape": plan.get("query_shape"),
        "entity_intersection": plan.get("entity_intersection"),
        "chart_spec": plan.get("chart_spec"),
    }


def _build_and_validate(
    *,
    slots: Any,
    question: str,
    bundle: ConfigBundle,
    chart_suppressed: bool,
    preferred_chart_type: Optional[str],
) -> tuple[YamlDict, YamlDict]:
    plan = build_plan_from_slots(
        slots,
        bundle=bundle,
        chart_suppressed=chart_suppressed,
        preferred_chart_type=preferred_chart_type,
    )
    plan["normalized_sub_request"] = question
    validation = _validate_and_repair_plan(
        plan,
        policy=_build_validation_policy(bundle),
    )
    return plan, validation


def _validation_debug(validation: YamlDict) -> YamlDict:
    return {
        "valid": bool(validation.get("valid")),
        "errors": list(validation.get("errors", []) or []),
        "blocking_errors": list(validation.get("blocking_errors", []) or []),
        "repairs": list(validation.get("repairs", []) or []),
    }


def _should_try_llm_after_deterministic(
    *,
    question: str,
    detected_entities: Optional[YamlDict],
    validation: YamlDict,
    debug: YamlDict,
) -> bool:
    if "llm_slot_extraction" in debug:
        return False
    if not validation.get("valid"):
        return True

    metrics = (detected_entities or {}).get("metrics") or []
    if isinstance(metrics, list) and len({str(item) for item in metrics}) > 1:
        return True

    text = str(question or "").lower()
    complex_markers = (
        "percentage increase",
        "percentage decrease",
        "percent increase",
        "percent decrease",
        "biggest increase",
        "biggest decrease",
        "increase in",
        "decrease in",
    )
    return any(marker in text for marker in complex_markers)
