"""
AI Quant Trading Platform — Market Data Service Models

Common data structures shared across the market data layer.
All data flowing through the system uses these typed models.

Why Pydantic models:
- Runtime validation (catch bad data before it reaches the strategy)
- Serialization/deserialization (JSON, dict, DB)
- Type hints for IDE support
- Documentation via field descriptions
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Timeframe(str, Enum):
    """Supported bar timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN = "1M"


class QualityStatus(str, Enum):
    """Data quality status."""
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"
    STALE = "stale"
    MISSING = "missing"


class Bar(BaseModel):
    """OHLCV bar (candlestick) data."""
    symbol: str
    timeframe: Timeframe
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    trade_count: int | None = None
    source: str = ""
    quality_status: QualityStatus = QualityStatus.VALID

    @property
    def mid(self) -> float:
        """Mid price of the bar."""
        return (self.high + self.low) / 2

    @property
    def body(self) -> float:
        """Body size (absolute difference between open and close)."""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """Total range (high - low)."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open


class Quote(BaseModel):
    """Bid/Ask quote data."""
    symbol: str
    time: datetime
    bid: float
    ask: float
    bid_size: int = 0
    ask_size: int = 0
    source: str = ""

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        mid = (self.bid + self.ask) / 2
        return self.spread / mid if mid > 0 else 0


class Trade(BaseModel):
    """Individual trade/tick data."""
    symbol: str
    time: datetime
    price: float
    size: int
    exchange: str = ""
    conditions: str = ""
    source: str = ""


class MarketSnapshot(BaseModel):
    """Point-in-time snapshot of an instrument's market state."""
    symbol: str
    time: datetime
    last_price: float
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    volume: int = 0
    vwap: float | None = None
    daily_change: float | None = None
    daily_change_pct: float | None = None


class CorporateAction(BaseModel):
    """Stock split, dividend, etc."""
    symbol: str
    action_type: str  # split, dividend, merger
    ex_date: datetime
    details: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
