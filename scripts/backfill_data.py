"""
AI Quant Trading Platform — Historical Data Backfill Script

Downloads and stores historical market data from Alpaca.

Usage:
    python scripts/backfill_data.py --symbols AAPL,MSFT,GOOGL --days 365 --timeframe 1d
    python scripts/backfill_data.py --universe sp500 --days 90 --timeframe 1h
    python scripts/backfill_data.py --symbols SPY --days 1825 --timeframe 1d  # 5 years

Features:
- Respects Alpaca API rate limits
- Validates all data through Data Quality Engine
- Bulk inserts for performance
- Progress logging
- Resumable (skips existing data via ON CONFLICT DO NOTHING)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings
from core.logging import setup_logging, get_logger
from core.database import init_database, close_database
from core.redis_client import init_redis, close_redis
from services.market_data.alpaca_adapter import AlpacaAdapter
from services.market_data.ingestor import DataIngestor
from services.market_data.models import Timeframe

logger = get_logger("backfill")

# ── Common Symbol Lists ────────────────────
# These are starting points. In production, use instrument table from DB.

CORE_ETFS = ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "XLK", "XLF", "XLE",
             "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB", "GLD",
             "SLV", "TLT", "HYG", "VIX"]

TOP_50_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
    "UNH", "JNJ", "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
    "ACN", "TMO", "CRM", "ABT", "DHR", "NFLX", "ADBE", "AMD", "TXN",
    "NEE", "PM", "UPS", "RTX", "LOW", "HON", "INTC", "ORCL", "QCOM",
    "BA", "GS", "CAT", "AMAT", "SBUX",
]


def parse_timeframe(tf_str: str) -> Timeframe:
    """Parse timeframe string to enum."""
    mapping = {
        "1m": Timeframe.M1, "5m": Timeframe.M5, "15m": Timeframe.M15,
        "30m": Timeframe.M30, "1h": Timeframe.H1, "4h": Timeframe.H4,
        "1d": Timeframe.D1, "1w": Timeframe.W1, "1M": Timeframe.MN,
    }
    if tf_str not in mapping:
        raise ValueError(f"Invalid timeframe: {tf_str}. Valid: {list(mapping.keys())}")
    return mapping[tf_str]


async def backfill_symbol(
    adapter: AlpacaAdapter,
    ingestor: DataIngestor,
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
) -> dict[str, int]:
    """Backfill historical data for a single symbol."""
    logger.info(
        "backfill_start",
        symbol=symbol,
        timeframe=timeframe.value,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    try:
        bars = await adapter.get_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        if not bars:
            logger.warning("backfill_no_data", symbol=symbol)
            return {"symbol": symbol, "ingested": 0, "rejected": 0}

        result = await ingestor.ingest_bars_bulk(bars)

        logger.info(
            "backfill_complete",
            symbol=symbol,
            fetched=len(bars),
            ingested=result["ingested"],
            rejected=result["rejected"],
        )

        return {"symbol": symbol, **result}

    except Exception as e:
        logger.error("backfill_error", symbol=symbol, error=str(e))
        return {"symbol": symbol, "ingested": 0, "rejected": 0, "error": str(e)}


async def main(args: argparse.Namespace) -> None:
    """Main backfill function."""
    setup_logging()

    # Determine symbols
    if args.universe == "etfs":
        symbols = CORE_ETFS
    elif args.universe == "top50":
        symbols = TOP_50_STOCKS
    elif args.universe == "all":
        symbols = CORE_ETFS + TOP_50_STOCKS
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = CORE_ETFS
        logger.info("using_default_universe", universe="etfs")

    timeframe = parse_timeframe(args.timeframe)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    logger.info(
        "backfill_config",
        symbols_count=len(symbols),
        timeframe=timeframe.value,
        days=args.days,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    # Initialize infrastructure
    await init_database()
    await init_redis()

    # Initialize adapter
    adapter = AlpacaAdapter()
    await adapter.connect()

    ingestor = DataIngestor()

    # Backfill each symbol
    total_ingested = 0
    total_rejected = 0
    errors = []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"backfill_progress", current=i, total=len(symbols), symbol=symbol)

        result = await backfill_symbol(adapter, ingestor, symbol, timeframe, start, end)
        total_ingested += result.get("ingested", 0)
        total_rejected += result.get("rejected", 0)
        if "error" in result:
            errors.append(result)

        # Small delay between symbols to respect rate limits
        await asyncio.sleep(0.5)

    # Summary
    logger.info(
        "backfill_summary",
        symbols=len(symbols),
        total_ingested=total_ingested,
        total_rejected=total_rejected,
        errors=len(errors),
    )

    if errors:
        logger.warning("backfill_errors", errors=errors)

    # Cleanup
    await adapter.disconnect()
    await close_redis()
    await close_database()

    print(f"\n✅ Backfill complete:")
    print(f"   Symbols: {len(symbols)}")
    print(f"   Bars ingested: {total_ingested}")
    print(f"   Bars rejected: {total_rejected}")
    print(f"   Errors: {len(errors)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical market data")
    parser.add_argument(
        "--symbols", type=str, default="",
        help="Comma-separated symbols (e.g., AAPL,MSFT,GOOGL)"
    )
    parser.add_argument(
        "--universe", type=str, choices=["etfs", "top50", "all"],
        default="", help="Predefined symbol universe"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d",
        help="Bar timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)"
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Number of days of historical data"
    )

    parsed_args = parser.parse_args()
    asyncio.run(main(parsed_args))
