"""
Backtesting service package.

Provides event-driven simulation, realistic cost modeling, and quantitative performance analytics.
"""

from services.backtesting.cost_model import CostModel
from services.backtesting.metrics import MetricsCalculator, PerformanceMetrics
from services.backtesting.portfolio import (
    SimulatedPortfolio,
    SimulatedPosition,
    SimulatedTrade,
)
from services.backtesting.engine import (
    EventDrivenBacktester,
    BacktestResult,
)

__all__ = [
    "CostModel",
    "MetricsCalculator",
    "PerformanceMetrics",
    "SimulatedPortfolio",
    "SimulatedPosition",
    "SimulatedTrade",
    "EventDrivenBacktester",
    "BacktestResult",
]
