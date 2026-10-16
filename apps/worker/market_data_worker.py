"""
AI Quant Trading Platform — Market Data Worker

Background worker that:
1. Connects to Alpaca WebSocket for real-time data
2. Routes incoming bars/quotes through Data Quality → Ingestor
3. Monitors connection health
4. Reconnects automatically on failures

This worker runs continuously as a background process.

Usage:
    python -m apps.worker.market_data_worker
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.config import get_settings
from core.logging import setup_logging, get_logger
from core.database import init_database, close_database
from core.redis_client import init_redis, close_redis
from core.event_bus import publish_event, EventType
from services.market_data.alpaca_adapter import AlpacaAdapter
from services.market_data.ingestor import DataIngestor
from services.market_data.models import Bar, Quote, Timeframe

logger = get_logger("market_data_worker")


class MarketDataWorker:
    """
    Real-time market data ingestion worker.
    
    Connects to Alpaca WebSocket and routes all data through
    the validation pipeline before storage and event publishing.
    """

    def __init__(self) -> None:
        self._adapter = AlpacaAdapter()
        self._ingestor = DataIngestor()
        self._running = False
        self._symbols: list[str] = []

    async def start(self, symbols: list[str] | None = None) -> None:
        """
        Start the market data worker.
        
        Args:
            symbols: Symbols to subscribe to. If None, loads from config/DB.
        """
        setup_logging()
        settings = get_settings()

        logger.info(
            "worker_starting",
            trading_mode=settings.trading_mode.value,
        )

        # Initialize infrastructure
        await init_database()
        await init_redis()

        # Default symbols if none provided
        self._symbols = symbols or [
            "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
            "TSLA", "AMD", "JPM", "V", "JNJ", "UNH",
        ]

        # Connect to Alpaca
        await self._adapter.connect()

        # Subscribe to real-time data
        await self._adapter.subscribe_bars(
            symbols=self._symbols,
            timeframe=Timeframe.M1,
            callback=self._on_bar,
        )

        await self._adapter.subscribe_quotes(
            symbols=self._symbols,
            callback=self._on_quote,
        )

        self._running = True
        logger.info(
            "worker_started",
            symbols_count=len(self._symbols),
            symbols=self._symbols[:10],  # Log first 10
        )

        # Publish system event
        await publish_event(
            event_type=EventType.SYSTEM_HEALTH_CHANGED,
            data={
                "service": "market_data_worker",
                "status": "healthy",
                "symbols": len(self._symbols),
            },
            source="market_data_worker",
        )

        # Keep running until stopped
        try:
            while self._running:
                await asyncio.sleep(30)

                # Periodic health check
                health = await self._adapter.health_check()
                stats = self._ingestor.get_stats()

                logger.info(
                    "worker_heartbeat",
                    adapter_status=health["status"],
                    ws_connected=health.get("websocket", {}).get("connected"),
                    ingested=stats["ingested"],
                    rejected=stats["rejected"],
                )

                # Check for stale data
                stale = self._ingestor._quality.get_stale_symbols(self._symbols)
                if stale:
                    logger.warning("stale_symbols_detected", symbols=stale[:5])

        except asyncio.CancelledError:
            logger.info("worker_cancelled")

        await self.stop()

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False
        await self._adapter.disconnect()
        await close_redis()
        await close_database()
        logger.info("worker_stopped")

    async def _on_bar(self, bar: Bar) -> None:
        """Handle incoming real-time bar."""
        success = await self._ingestor.ingest_bar(bar)
        if success:
            logger.debug(
                "rt_bar_ingested",
                symbol=bar.symbol,
                close=bar.close,
                volume=bar.volume,
            )

    async def _on_quote(self, quote: Quote) -> None:
        """Handle incoming real-time quote."""
        await self._ingestor.ingest_quote(quote)


async def run() -> None:
    """Entry point for the worker."""
    worker = MarketDataWorker()

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("shutdown_signal_received")
        asyncio.create_task(worker.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    await worker.start()


if __name__ == "__main__":
    asyncio.run(run())
