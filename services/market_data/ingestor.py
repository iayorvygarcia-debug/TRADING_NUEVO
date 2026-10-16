"""
AI Quant Trading Platform — Market Data Ingestion Pipeline

Orchestrates the flow:
    Market Data Provider → Data Quality → Storage → Event Bus

This is the central pipeline that:
1. Receives data from any MarketDataProvider (Alpaca, Polygon, etc.)
2. Validates it through the Data Quality Engine
3. Stores valid data in TimescaleDB
4. Publishes events to the Event Bus for downstream consumers

Architecture:
    AlpacaAdapter.on_bar() 
        → DataIngestor.ingest_bar()
            → DataQualityEngine.validate_bar()
                → IF VALID: store in DB + publish BarClosed event
                → IF INVALID: log + reject + publish quality alert

CRITICAL: No data reaches strategies or models without passing through here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.logging import get_logger
from core.event_bus import publish_event, EventType
from core.database import get_session
from services.market_data.models import Bar, Quote, Timeframe, QualityStatus
from services.data_quality.validators import DataQualityEngine

logger = get_logger("data_ingestor")


class DataIngestor:
    """
    Central data ingestion pipeline.
    
    Connects market data providers to storage and event bus.
    All data must flow through here.
    """

    def __init__(self, quality_engine: DataQualityEngine | None = None) -> None:
        self._quality = quality_engine or DataQualityEngine()
        self._ingested_count = 0
        self._rejected_count = 0

    # ── Bar Ingestion ───────────────────────

    async def ingest_bar(self, bar: Bar) -> bool:
        """
        Ingest a bar: validate → store → publish event.
        
        Args:
            bar: The bar to ingest.
        
        Returns:
            True if bar was stored successfully, False if rejected.
        """
        # Step 1: Validate
        report = self._quality.validate_bar(bar)

        if not report.is_valid:
            self._rejected_count += 1
            logger.warning(
                "bar_rejected",
                symbol=bar.symbol,
                timeframe=bar.timeframe.value,
                time=bar.time.isoformat(),
                issues=report.issues,
            )
            return False

        # Update bar quality status
        bar.quality_status = report.status

        # Step 2: Store in database
        try:
            await self._store_bar(bar)
        except Exception as e:
            logger.error(
                "bar_store_failed",
                symbol=bar.symbol,
                error=str(e),
            )
            return False

        # Step 3: Publish event to Event Bus
        try:
            await publish_event(
                event_type=EventType.BAR_CLOSED,
                data={
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe.value,
                    "time": bar.time.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                    "quality": report.status.value,
                },
                source="data_ingestor",
            )
        except Exception as e:
            # Event publish failure is not critical — data is already stored
            logger.warning("bar_event_publish_failed", error=str(e))

        self._ingested_count += 1

        if report.warnings:
            logger.debug(
                "bar_ingested_with_warnings",
                symbol=bar.symbol,
                warnings=report.warnings,
            )

        return True

    async def _store_bar(self, bar: Bar) -> None:
        """Store a bar in TimescaleDB."""
        from sqlalchemy import text

        async with get_session() as session:
            await session.execute(
                text("""
                    INSERT INTO bars (time, symbol, timeframe, open, high, low, close,
                                      volume, vwap, trade_count, source, quality_status)
                    VALUES (:time, :symbol, :timeframe, :open, :high, :low, :close,
                            :volume, :vwap, :trade_count, :source, :quality_status)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "time": bar.time,
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe.value,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                    "trade_count": bar.trade_count,
                    "source": bar.source,
                    "quality_status": bar.quality_status.value,
                },
            )

    # ── Bulk Bar Ingestion ──────────────────

    async def ingest_bars_bulk(self, bars: list[Bar]) -> dict[str, int]:
        """
        Ingest multiple bars efficiently.
        
        Used by backfill scripts and historical data loading.
        Validates each bar but uses bulk insert for performance.
        
        Returns:
            Dict with ingested/rejected counts.
        """
        valid_bars: list[Bar] = []
        rejected = 0

        for bar in bars:
            report = self._quality.validate_bar(bar)
            if report.is_valid:
                bar.quality_status = report.status
                valid_bars.append(bar)
            else:
                rejected += 1

        if valid_bars:
            try:
                await self._store_bars_bulk(valid_bars)
            except Exception as e:
                logger.error("bulk_store_failed", error=str(e), count=len(valid_bars))
                return {"ingested": 0, "rejected": rejected, "error": str(e)}

        self._ingested_count += len(valid_bars)
        self._rejected_count += rejected

        logger.info(
            "bulk_ingest_complete",
            ingested=len(valid_bars),
            rejected=rejected,
            total=len(bars),
        )

        return {"ingested": len(valid_bars), "rejected": rejected}

    async def _store_bars_bulk(self, bars: list[Bar]) -> None:
        """Bulk insert bars into TimescaleDB."""
        from sqlalchemy import text

        if not bars:
            return

        async with get_session() as session:
            # Use executemany for bulk insert
            await session.execute(
                text("""
                    INSERT INTO bars (time, symbol, timeframe, open, high, low, close,
                                      volume, vwap, trade_count, source, quality_status)
                    VALUES (:time, :symbol, :timeframe, :open, :high, :low, :close,
                            :volume, :vwap, :trade_count, :source, :quality_status)
                    ON CONFLICT DO NOTHING
                """),
                [
                    {
                        "time": b.time,
                        "symbol": b.symbol,
                        "timeframe": b.timeframe.value,
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                        "vwap": b.vwap,
                        "trade_count": b.trade_count,
                        "source": b.source,
                        "quality_status": b.quality_status.value,
                    }
                    for b in bars
                ],
            )

    # ── Quote Ingestion ─────────────────────

    async def ingest_quote(self, quote: Quote) -> bool:
        """Ingest a quote: validate → store → publish."""
        report = self._quality.validate_quote(quote)

        if not report.is_valid:
            return False

        try:
            await self._store_quote(quote)
        except Exception as e:
            logger.error("quote_store_failed", symbol=quote.symbol, error=str(e))
            return False

        await publish_event(
            event_type=EventType.QUOTE_UPDATED,
            data={
                "symbol": quote.symbol,
                "time": quote.time.isoformat(),
                "bid": quote.bid,
                "ask": quote.ask,
                "spread": quote.spread,
            },
            source="data_ingestor",
        )

        return True

    async def _store_quote(self, quote: Quote) -> None:
        """Store a quote in TimescaleDB."""
        from sqlalchemy import text

        async with get_session() as session:
            await session.execute(
                text("""
                    INSERT INTO quotes (time, symbol, bid, ask, bid_size, ask_size,
                                        spread, source)
                    VALUES (:time, :symbol, :bid, :ask, :bid_size, :ask_size,
                            :spread, :source)
                """),
                {
                    "time": quote.time,
                    "symbol": quote.symbol,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "spread": quote.spread,
                    "source": quote.source,
                },
            )

    # ── Stats & Health ──────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get ingestion pipeline stats."""
        return {
            "ingested": self._ingested_count,
            "rejected": self._rejected_count,
            "total": self._ingested_count + self._rejected_count,
            "reject_rate": (
                round(self._rejected_count / (self._ingested_count + self._rejected_count), 4)
                if (self._ingested_count + self._rejected_count) > 0
                else 0.0
            ),
            "quality": self._quality.get_stats(),
        }
