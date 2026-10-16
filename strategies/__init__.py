"""
Trading strategies package.

Exports base interfaces and concrete strategies:
- TrendFollowingStrategy
- MeanReversionStrategy
- BreakoutStrategy
- VWAPReversionStrategy
- MultiTimeframeConfluenceStrategy
"""

from strategies.base import (
    SignalDirection,
    Strategy,
    StrategySignal,
)
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.multi_timeframe_confluence import MultiTimeframeConfluenceStrategy

__all__ = [
    "SignalDirection",
    "Strategy",
    "StrategySignal",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "VWAPReversionStrategy",
    "MultiTimeframeConfluenceStrategy",
]
