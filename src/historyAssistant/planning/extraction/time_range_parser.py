from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

YamlDict = Dict[str, Any]

_TIME_TOKEN = r"(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)"
_TIME_RANGE_RE = re.compile(r"(?i)\b(?:between|from)\s+" + _TIME_TOKEN + r"\s+(?:and|to)\s+" + _TIME_TOKEN + r"\b")
_AROUND_TIME_RE = re.compile(r"(?i)\baround\s+(\d{1,2}:\d{2})\b")
_EXACT_TIME_RE = re.compile(r"(?i)\bat\s+(\d{1,2}:\d{2})\b")
_HOUR_BUCKET_RE = re.compile(r"(?i)\bat\s+(\d{1,2})(?!\s*:)\b")


def extract_time_range(
    question: str,
    date_rules_config: Optional[YamlDict] = None,
    *,
    skip_day_part: bool = False,
) -> Optional[Dict[str, str]]:
    match = _TIME_RANGE_RE.search(question or "")
    if match:
        return {"start_time": parse_time_token(match.group(1)), "end_time": parse_time_token(match.group(2))}

    if not skip_day_part:
        label = detect_day_part(question, date_rules_config)
        cfg = configured_day_parts(date_rules_config).get(label) if label else None
        if cfg:
            return {"start_time": str(cfg["start_time"]), "end_time": str(cfg["end_time"])}

    around = _AROUND_TIME_RE.search(question or "")
    if around:
        return {
            "start_time": shift_clock_minutes(around.group(1), -10),
            "end_time": shift_clock_minutes(around.group(1), 10),
        }

    hour_bucket = _HOUR_BUCKET_RE.search(question or "")
    if hour_bucket:
        hour = int(hour_bucket.group(1))
        if 0 <= hour <= 23:
            return {"start_time": f"{hour:02d}:00", "end_time": f"{hour:02d}:59"}

    exact = _EXACT_TIME_RE.search(question or "")
    return {"start_time": exact.group(1), "end_time": exact.group(1)} if exact else None


def detect_day_part(question: str, date_rules_config: Optional[YamlDict] = None) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if not text:
        return None
    candidates = [
        (label, str(alias or "").strip().lower())
        for label, cfg in configured_day_parts(date_rules_config).items()
        for alias in cfg.get("aliases", []) or []
        if str(alias or "").strip()
    ]
    for label, alias in sorted(candidates, key=lambda item: -len(item[1])):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text):
            return label
    return None


def configured_day_parts(date_rules_config: Optional[YamlDict]) -> Dict[str, YamlDict]:
    result: Dict[str, YamlDict] = {}
    for name, cfg in ((date_rules_config or {}).get("day_parts", {}) or {}).items():
        if not isinstance(cfg, dict):
            continue
        start, end = cfg.get("start_time"), cfg.get("end_time")
        if not start or not end:
            continue
        aliases = [str(item).strip() for item in cfg.get("aliases", []) or [] if str(item).strip()]
        result[str(name)] = {"start_time": str(start), "end_time": str(end), "aliases": aliases or [str(name)]}
    return result


def parse_time_token(token: str) -> str:
    text = token.strip().lower().replace(" ", "")
    am_pm = None
    if text.endswith("am"):
        am_pm, text = "am", text[:-2]
    elif text.endswith("pm"):
        am_pm, text = "pm", text[:-2]

    if ":" in text:
        hour_s, minute_s = text.split(":", 1)
        hour, minute = int(hour_s), int(minute_s)
    else:
        hour, minute = int(text), 0

    if am_pm == "am" and hour == 12:
        hour = 0
    elif am_pm == "pm" and hour != 12:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def shift_clock_minutes(time_text: str, delta_minutes: int) -> str:
    return (datetime.strptime(time_text, "%H:%M") + timedelta(minutes=delta_minutes)).strftime("%H:%M")
