"""
AI Quant Trading Platform — Risk Engine Data Models.

Typed models for risk levels, state tracking, and pre-trade validations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any


class RiskLevel(IntEnum):
    """
    Tiered risk levels for the autonomous Circuit Breaker.
    Higher values represent more restrictive states.
    """
    NORMAL = 0          # Optimal operating conditions, all actions permitted
    WARNING = 1         # Elevated volatility/drawdown, alerts triggered, reduced sizing
    STOP_NEW = 2        # Stop opening new positions, existing positions managed
    DE_RISK = 3         # Cut overall portfolio exposure by 50%, tighten stops
    EMERGENCY = 4       # Kill switch engaged, cancel all orders, halt all trading


@dataclass(slots=True)
class PreTradeCheckResult:
    """Outcome of a pre-trade deterministic risk evaluation."""
    approved: bool
    symbol: str
    direction: str
    requested_quantity: float
    approved_quantity: float
    reasons: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    circuit_breaker_level: RiskLevel = RiskLevel.NORMAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def was_scaled_down(self) -> bool:
        return self.approved and (self.approved_quantity < self.requested_quantity)


@dataclass
class PortfolioRiskState:
    """Current risk snapshot of the account/portfolio."""
    equity: float
    peak_equity: float
    cash: float
    daily_starting_equity: float
    open_positions_count: int = 0
    daily_trades_count: int = 0
    consecutive_losses: int = 0
    sector_exposures: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def daily_pnl(self) -> float:
        return self.equity - self.daily_starting_equity

    @property
    def daily_pnl_pct(self) -> float:
        if self.daily_starting_equity <= 0:
            return 0.0
        return (self.daily_pnl / self.daily_starting_equity) * 100.0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        dd = (self.equity - self.peak_equity) / self.peak_equity
        return abs(min(0.0, dd)) * 100.0

    @property
    def daily_peak_profit(self) -> float:
        """Peak profit achieved today above starting equity."""
        return max(0.0, self.peak_equity - self.daily_starting_equity)

    @property
    def profit_giveback_pct(self) -> float:
        """Percentage of daily peak profit given back if in profit."""
        peak_profit = self.daily_peak_profit
        if peak_profit <= 0.0:
            return 0.0
        current_profit = self.daily_pnl
        giveback = peak_profit - current_profit
        return max(0.0, (giveback / peak_profit) * 100.0)
