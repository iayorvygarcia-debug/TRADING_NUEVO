"""
AI Quant Trading Platform — Technical Analysis Service.

Provides mathematical indicators, price action patterns, multi-timeframe aggregation,
and the quantitative feature engine.
"""

from services.technical_analysis.indicators import (
    TechnicalIndicators,
    calculate_all_indicators,
)
from services.technical_analysis.patterns import (
    CandlePatternDetector,
    PatternResult,
)
from services.technical_analysis.multi_timeframe import (
    MultiTimeframeAggregator,
    TimeframeAlignment,
)
from services.technical_analysis.feature_engine import (
    FeatureEngine,
    FeatureVector,
)

from services.technical_analysis.feature_store import (
    FeatureStore,
    get_feature_store,
)

__all__ = [
    "TechnicalIndicators",
    "calculate_all_indicators",
    "CandlePatternDetector",
    "PatternResult",
    "MultiTimeframeAggregator",
    "TimeframeAlignment",
    "FeatureEngine",
    "FeatureVector",
    "FeatureStore",
    "get_feature_store",
]
