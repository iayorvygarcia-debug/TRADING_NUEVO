"""
AI Quant Trading Platform — Market Data Provider Interface

Abstract base class for all market data providers.
This is the adapter pattern that prevents coupling to any single provider.

Architecture:
    MarketDataProvider (interface)
    ├── AlpacaAdapter      (FASE 2 - primary)
    ├── PolygonAdapter     (future - secondary/backup)
    └── HistoricalAdapter  (for backtesting from DB)

Why this interface exists:
- Provider independence (can switch Alpaca → IBKR without rewriting strategies)
- Testing (mock adapter for unit tests)
- Multi-source (aggregate data from multiple providers)
- Failover (if primary goes down, fall back to secondary)

Each adapter must implement:
1. connect/disconnect lifecycle
2. Subscribe to real-time data (WebSocket preferred)
3. Fetch historical bars
4. Get current quotes/snapshots
5. Report its own health status
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, Callable, Coroutine, Any

from services.market_data.models import Bar, Quote, Trade, MarketSnapshot, Timeframe


# Callback types for real-time data
BarCallback = Callable[[Bar], Coroutine[Any, Any, None]]
QuoteCallback = Callable[[Quote], Coroutine[Any, Any, None]]
TradeCallback = Callable[[Trade], Coroutine[Any, Any, None]]


class MarketDataProvider(ABC):
    """
    Abstract interface for market data providers.
    
    All concrete adapters (Alpaca, Polygon, etc.) must implement this.
    The rest of the platform interacts only with this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'alpaca', 'polygon')."""
        ...

    @property
    @abstractmethod
    def supports_websocket(self) -> bool:
        """Whether this provider supports WebSocket streaming."""
        ...

    # ── Lifecycle ───────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """
        Initialize connection to the data provider.
        
        For WebSocket providers, this establishes the WS connection.
        For REST-only providers, this validates the API key.
        
        Raises:
            DataFeedDisconnectedError: If connection fails.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connections and resources."""
        ...

    # ── Real-time Subscriptions ─────────────

    @abstractmethod
    async def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        callback: BarCallback,
    ) -> None:
        """
        Subscribe to real-time bar updates via WebSocket.
        
        Args:
            symbols: List of ticker symbols to subscribe to.
            timeframe: Bar timeframe.
            callback: Async function called for each new bar.
        """
        ...

    @abstractmethod
    async def subscribe_quotes(
        self,
        symbols: list[str],
        callback: QuoteCallback,
    ) -> None:
        """Subscribe to real-time quote updates."""
        ...

    @abstractmethod
    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: TradeCallback,
    ) -> None:
        """Subscribe to real-time trade updates."""
        ...

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        ...

    # ── Historical Data ─────────────────────

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Bar]:
        """
        Fetch historical bars.
        
        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Start datetime (inclusive).
            end: End datetime (inclusive). Defaults to now.
            limit: Max number of bars to return.
        
        Returns:
            List of Bar objects, sorted by time ascending.
        """
        ...

    @abstractmethod
    async def get_bars_stream(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime | None = None,
    ) -> AsyncGenerator[Bar, None]:
        """
        Stream historical bars (for large datasets).
        
        Yields bars one at a time to avoid loading entire history into memory.
        Used by backtesting engine and data backfill scripts.
        """
        ...
        yield  # type: ignore  # pragma: no cover

    # ── Snapshots ───────────────────────────

    @abstractmethod
    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """Get current market snapshot for a symbol."""
        ...

    @abstractmethod
    async def get_snapshots(self, symbols: list[str]) -> dict[str, MarketSnapshot]:
        """Get market snapshots for multiple symbols."""
        ...

    # ── Quote ───────────────────────────────

    @abstractmethod
    async def get_latest_quote(self, symbol: str) -> Quote | None:
        """Get the latest bid/ask quote for a symbol."""
        ...

    # ── Health ──────────────────────────────

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check provider health.
        
        Returns:
            Dict with at minimum: {"status": "healthy"|"unhealthy", ...}
        """
        ...
