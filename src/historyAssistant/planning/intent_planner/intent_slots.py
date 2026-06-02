from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

YamlDict = Dict[str, Any]


@dataclass(frozen=True)
class RankingSlots:
    detected: bool = False
    direction: Optional[str] = None
    limit: Optional[int] = None
    field: Optional[str] = None

    def to_dict(self) -> YamlDict:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonSlots:
    comparison_type: str
    date_rule: Optional[str] = None
    date_range: Optional[YamlDict] = None
    date_params: Optional[YamlDict] = None
    filters: List[YamlDict] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    preferred_direction: Optional[str] = None
    reason: str = ""

    def to_compare_to(self) -> YamlDict:
        return {
            "date_rule": self.date_rule,
            "date_range": self.date_range,
            "date_params": self.date_params,
            "filters": list(self.filters),
        }

    def to_dict(self) -> YamlDict:
        return asdict(self)


@dataclass(frozen=True)
class IntentSlots:
    source_id: str
    target_table: str
    answer_shape: str = "analytic_summary"
    request_type: str = "qa"
    metric: Optional[str] = None
    aggregation: Optional[str] = None
    metric_params: YamlDict = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    filters: List[YamlDict] = field(default_factory=list)
    source_fields: List[str] = field(default_factory=list)
    narration_fields: YamlDict = field(default_factory=dict)
    order_by: Optional[YamlDict] = None
    limit: Optional[int] = None
    projection_field: Optional[str] = None
    projection_mode: Optional[str] = None
    date_rule: Optional[str] = None
    date_range: Optional[YamlDict] = None
    date_params: Optional[YamlDict] = None
    time_range: Optional[YamlDict] = None
    comparison: Optional[ComparisonSlots] = None
    ranking: RankingSlots = field(default_factory=RankingSlots)
    bucket_interval: Optional[str] = None
    bucket_summary: Optional[str] = None
    confidence: float = 0.0
    extraction_path: str = "deterministic"
    reason: str = ""

    def to_dict(self) -> YamlDict:
        data = asdict(self)
        data["comparison"] = self.comparison.to_dict() if self.comparison else None
        data["ranking"] = self.ranking.to_dict()
        return data


@dataclass(frozen=True)
class SlotExtractionResult:
    used: bool
    slots: Optional[IntentSlots] = None
    reason: str = ""
    debug: YamlDict = field(default_factory=dict)

    def to_debug_dict(self) -> YamlDict:
        return {
            "used": self.used,
            "reason": self.reason,
            "slots": self.slots.to_dict() if self.slots else None,
            **dict(self.debug),
        }
