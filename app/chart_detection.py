from __future__ import annotations

import re
from typing import Optional

# keywords signal for when user wants a chart or visualization.
_CHART_INTENT_RE = re.compile(
    r"\b(?:"
    r"chart|graph|plot|visualize|visualization|diagram"
    r"|bar\s*chart|line\s*chart|pie\s*chart|scatter\s*plot"
    r"|show\s+(?:me\s+)?(?:a\s+)?(?:chart|graph|plot|visualization)"
    r")\b",
    re.IGNORECASE,
)

# extract a specific chart type from question.
_CHART_TYPE_RE = re.compile(
    r"\b(bar|horizontal[_ ]bar|stacked[_ ]bar|line|pie|scatter|area|histogram|dot[_ ]plot)"
    r"(?:\s*(?:chart|graph|plot|diagram|visualization))?\b",
    re.IGNORECASE,
)

_CHART_SUPPRESSION_RE = re.compile(
    r"\b(?:no|without|exclude|hide|skip)\s+(?:chart|graph|plot|visualization)s?\b"
    r"|\b(?:text|answer|table)\s+only\b"
    r"|\bdo\s+not\s+(?:chart|graph|plot|visualize)\b"
    r"|\bdon't\s+(?:chart|graph|plot|visualize)\b",
    re.IGNORECASE,
)

_TYPE_NORMALIZE = {
    "bar": "bar",
    "horizontal_bar": "horizontal_bar",
    "horizontal bar": "horizontal_bar",
    "stacked_bar": "stacked_bar",
    "stacked bar": "stacked_bar",
    "line": "line",
    "pie": "pie",
    "scatter": "scatter",
    "area": "area",
    "histogram": "histogram",
    "dot_plot": "dot_plot",
    "dot plot": "dot_plot",
}


def detect_chart_intent(question: str) -> bool:
    if suppress_chart_intent(question):
        return False
    return bool(_CHART_INTENT_RE.search(question or ""))


def suppress_chart_intent(question: str) -> bool:
    return bool(_CHART_SUPPRESSION_RE.search(question or ""))


def extract_preferred_chart_type(question: str) -> Optional[str]:
    match = _CHART_TYPE_RE.search(question or "")
    if not match:
        return None
    raw = match.group(1).lower()
    return _TYPE_NORMALIZE.get(raw)
