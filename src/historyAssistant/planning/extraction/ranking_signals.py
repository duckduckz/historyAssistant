from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_DESC_WORDS = [
    "top", "most", "highest", "longest", "slowest",
    "worst", "busiest", "largest", "greatest", "maximum", "max",
]

_ASC_WORDS = [
    "bottom", "fewest", "lowest", "shortest", "fastest",
    "best", "quietest", "smallest", "least", "minimum", "min",
]


def _word_alternation(words: List[str]) -> str:
    return "|".join(rf"\b{re.escape(w)}\b" for w in words)


_DESC_RE = re.compile(_word_alternation(_DESC_WORDS), re.IGNORECASE)
_ASC_RE = re.compile(_word_alternation(_ASC_WORDS), re.IGNORECASE)

_LIMIT_KEYWORDS = [
    "most", "highest", "lowest", "fewest", "busiest",
    "slowest", "fastest", "longest", "shortest", "worst", "best",
]
_LIMIT_KW = "|".join(_LIMIT_KEYWORDS)

_LIMIT_RE = re.compile(
    r"(?i)"
    r"(?:"
    r"(?:top|bottom)\s*[-]?\s*(\d+)"  
    r"|"
    rf"(?:the\s+)?(\d+)\s+(?:{_LIMIT_KW})"  
    r"|"
    rf"(?:which|what(?:\s+are)?|show(?:\s+me)?(?:\s+the)?)\s+(\d+)\s+\w+(?:\s+\w+){{0,4}}\s+(?:had|have|with|by|were|was)?\s*(?:the\s+)?(?:{_LIMIT_KW})"  
    r")"
)


@dataclass(frozen=True)
class RankingSignal:
    detected: bool
    direction: Optional[str] = None  
    limit: Optional[int] = None     
    matched_phrases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "detected": self.detected,
            "direction": self.direction,
            "limit": self.limit,
            "matched_phrases": list(self.matched_phrases),
        }


def detect_ranking_signals(question: str) -> RankingSignal:
    if not question or not question.strip():
        return RankingSignal(detected=False)

    matched_phrases: List[str] = []
    direction: Optional[str] = None

    # check desc signals
    desc_matches = _DESC_RE.findall(question)
    asc_matches = _ASC_RE.findall(question)

    if desc_matches:
        matched_phrases.extend(m.lower() for m in desc_matches)
        direction = "desc"
    if asc_matches:
        matched_phrases.extend(m.lower() for m in asc_matches)
        # if both directions found, last one wins (rare edge case)
        direction = "asc" if not desc_matches else direction

    # extract explicit limit
    limit: Optional[int] = None
    limit_match = _LIMIT_RE.search(question)
    if limit_match:
        raw_limit = limit_match.group(1) or limit_match.group(2) or limit_match.group(3)
        if raw_limit:
            limit = int(raw_limit)

    if not matched_phrases:
        return RankingSignal(detected=False)

    return RankingSignal(
        detected=True,
        direction=direction,
        limit=limit,
        matched_phrases=matched_phrases,
    )


__all__ = ["RankingSignal", "detect_ranking_signals"]
