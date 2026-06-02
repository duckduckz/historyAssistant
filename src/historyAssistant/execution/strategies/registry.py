from __future__ import annotations

from historyAssistant.execution.strategies.average_count_per_group_strategy import AvgCountPerGroupStrategy
from historyAssistant.execution.strategies.base import AggregationStrategy
from historyAssistant.execution.strategies.basic_metric_strategies import AvgStrategy, CardinalityStrategy, CountStrategy
from historyAssistant.execution.strategies.data_quality_metric_strategies import DqCountStrategy, DqRateStrategy
from historyAssistant.execution.strategies.numeric_summary_strategy import NumericSummaryStrategy
from historyAssistant.execution.strategies.percentile_metric_strategy import PercentileStrategy
from historyAssistant.execution.strategies.raw_record_strategies import FirstNumericStrategy, RawRecordsStrategy
from historyAssistant.execution.strategies.threshold_metric_strategies import CountThresholdStrategy, RateThresholdStrategy
from historyAssistant.execution.strategies.time_bucket_strategy import TimeBucketStrategy
from historyAssistant.execution.strategies.workflow_room_switch_strategy import RoomSwitchWithinMinutesStrategy


STRATEGY_REGISTRY: dict[str, AggregationStrategy] = {
    "count": CountStrategy(),
    "avg": AvgStrategy(),
    "cardinality": CardinalityStrategy(),
    "median": PercentileStrategy(default_percent=50.0),
    "percentile": PercentileStrategy(),
    "rate_threshold": RateThresholdStrategy(),
    "count_threshold": CountThresholdStrategy(),
    "dq_rate": DqRateStrategy(),
    "dq_count": DqCountStrategy(mode="single"),
    "dq_count_any_true": DqCountStrategy(mode="any"),
    "dq_count_all_true": DqCountStrategy(mode="all"),
    "avg_count_per_group": AvgCountPerGroupStrategy(),
    "numeric_summary": NumericSummaryStrategy(),
    "raw_records": RawRecordsStrategy(),
    "first_numeric": FirstNumericStrategy(),
    "room_switch_within_minutes": RoomSwitchWithinMinutesStrategy(),
    "time_bucket_summary": TimeBucketStrategy(),
}


__all__ = ["STRATEGY_REGISTRY"]
