"""
AI Quant Trading Platform — Strategy Interface

Abstract base class for all trading strategies.

CRITICAL RULES:
1. A strategy produces SIGNALS, never ORDERS
2. Signals must include confidence, score, and reasons
3. The strategy does NOT know which broker is being used
4. The strategy does NOT have access to the Execution Engine
5. NO TRADE is a valid output

Architecture:
    Strategy → StrategySignal → Signal Engine → Risk → Execution → Broker

Each strategy must:
1. Analyze market data (bars, features, regime)
2. Determine if an opportunity exists
3. Return a typed signal with score + reasons
4. Or return None (= NO TRADE, which is perfectly valid)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class StrategySignal(BaseModel):
    """
    Output from a strategy. This is NOT an order.
    
    This signal must pass through:
    Signal Engine → Portfolio → Risk → Execution
    before any order is created.
    """
    strategy: str               # Name of the strategy that generated this
    symbol: str
    direction: SignalDirection
    score: float = Field(ge=0.0, le=1.0)        # 0-1 confidence score
    confidence: float = Field(ge=0.0, le=1.0)   # How confident the strategy is
    expected_return: float | None = None          # Expected return estimate
    risk_estimate: float | None = None            # Estimated risk (e.g., ATR-based)
    timeframe: str | None = None                  # Primary timeframe for this signal
    stop_suggestion: float | None = None          # Suggested stop-loss price
    target_suggestion: float | None = None        # Suggested take-profit price
    market_regime: str | None = None              # Market regime when signal was generated
    reasons: list[str] = Field(default_factory=list)  # Why the signal was generated
    metadata: dict[str, Any] = Field(default_factory=dict)  # Additional data
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_actionable(self) -> bool:
        """Whether this signal suggests action (not HOLD)."""
        return self.direction != SignalDirection.HOLD


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Subclasses implement specific logic (trend following, momentum, etc.)
    but all share the same interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (e.g., 'trend_following', 'momentum')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the strategy logic."""
        ...

    @property
    @abstractmethod
    def timeframes(self) -> list[str]:
        """Timeframes this strategy operates on (e.g., ['1d', '4h', '1h'])."""
        ...

    @abstractmethod
    async def analyze(
        self,
        symbol: str,
        features: dict[str, Any],
        market_regime: str | None = None,
    ) -> StrategySignal | None:
        """
        Analyze a symbol and optionally generate a signal.
        
        Args:
            symbol: Ticker symbol.
            features: Pre-computed features from Feature Engine.
            market_regime: Current market regime (if available).
        
        Returns:
            StrategySignal if an opportunity is detected.
            None if NO TRADE (which is a valid decision).
        """
        ...

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Get current strategy configuration.
        
        Returns all parameters for audit/reproducibility.
        """
        ...
