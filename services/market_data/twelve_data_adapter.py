"""
AI Quant Trading Platform — Twelve Data Adapter

Multi-purpose market data adapter for Twelve Data API.

Key capabilities:
- Pre-computed technical indicators (55+ indicators via API)
- Real-time and historical price data
- Multi-key rotation for maximizing free-tier limits

Key Rotation Strategy:
    5 keys × 800 credits/day = 4,000 credits/day
    5 keys × 8 req/min = 40 req/min aggregate
    
    Round-robin with per-key rate limiting and credit tracking.
    If a key hits its daily limit → skip to next key.
    If all keys exhausted → wait until midnight ET reset.

Credit costs (Twelve Data pricing):
    Time series:    1 credit/request
    Technical ind:  1 credit/request  
    Quote:          1 credit/request
    Real-time WS:   1 credit/symbol/min (not used, we use Alpaca WS)

API docs: https://twelvedata.com/docs

Why Twelve Data alongside Alpaca:
- Alpaca: Primary broker + real-time WebSocket data
- Twelve Data: Technical indicators API (saves local computation)
- Together: Best of both worlds
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone, date
from typing import Any

import httpx

from core.config import get_settings
from core.logging import get_logger

logger = get_logger("twelve_data")

# ── Twelve Data timeframe mapping ───────────
TD_TIMEFRAME_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1day", "1w": "1week", "1M": "1month",
}

# ── Available Indicators ────────────────────
# Twelve Data computes these server-side (no local TA-Lib needed)
AVAILABLE_INDICATORS = [
    # Trend
    "sma", "ema", "wma", "dema", "tema", "t3", "kama", "mama",
    "trima", "bbands", "sar", "supertrend", "ichimoku",
    # Momentum
    "rsi", "stoch", "stochrsi", "macd", "cci", "roc", "mom",
    "willr", "adx", "dx", "plus_di", "minus_di", "aroon",
    "aroonosc", "trix", "ppo", "apo", "ultosc", "mfi",
    # Volume
    "obv", "ad", "adosc", "vwap",
    # Volatility
    "atr", "natr", "stddev", "percent_b", "midprice",
    # Statistics
    "beta", "correl", "linearreg", "linearreg_angle",
    "linearreg_intercept", "linearreg_slope", "tsf",
]


class KeyRotator:
    """
    Manages rotation across multiple API keys.
    
    Features:
    - Round-robin key selection
    - Per-key rate limiting (8 req/min)
    - Per-key daily credit tracking (800/day)
    - Automatic skip when key is exhausted
    - Credit counter resets at midnight ET
    """

    def __init__(self, keys: list[str], rate_per_key: int = 8, credits_per_key: int = 800):
        if not keys:
            raise ValueError("At least one API key is required")

        self._keys = keys
        self._rate_per_key = rate_per_key
        self._credits_per_key = credits_per_key
        self._current_index = 0

        # Per-key tracking
        self._request_times: dict[int, list[float]] = defaultdict(list)
        self._daily_credits: dict[int, int] = defaultdict(int)
        self._credits_date: date = date.today()

        logger.info(
            "key_rotator_initialized",
            num_keys=len(keys),
            total_credits=len(keys) * credits_per_key,
            aggregate_rate=len(keys) * rate_per_key,
        )

    @property
    def num_keys(self) -> int:
        return len(self._keys)

    @property
    def total_credits_remaining(self) -> int:
        self._maybe_reset_daily()
        return sum(
            max(0, self._credits_per_key - self._daily_credits[i])
            for i in range(self.num_keys)
        )

    def _maybe_reset_daily(self) -> None:
        """Reset credit counters at midnight."""
        today = date.today()
        if today != self._credits_date:
            self._daily_credits.clear()
            self._credits_date = today
            logger.info("daily_credits_reset", date=today.isoformat())

    async def get_key(self) -> str:
        """
        Get the next available API key, respecting rate limits and credits.
        
        Returns:
            An API key string.
        
        Raises:
            RuntimeError if all keys are exhausted for the day.
        """
        self._maybe_reset_daily()

        # Try each key starting from current index
        for attempt in range(self.num_keys):
            idx = (self._current_index + attempt) % self.num_keys

            # Check daily credits
            if self._daily_credits[idx] >= self._credits_per_key:
                continue  # This key is exhausted for today

            # Check rate limit (8 req/min)
            now = time.monotonic()
            self._request_times[idx] = [
                t for t in self._request_times[idx] if now - t < 60
            ]

            if len(self._request_times[idx]) >= self._rate_per_key:
                # This key is rate-limited, try next
                continue

            # This key is available
            self._request_times[idx].append(now)
            self._daily_credits[idx] += 1
            self._current_index = (idx + 1) % self.num_keys  # Advance for next call
            return self._keys[idx]

        # All keys are either rate-limited or exhausted
        # Wait for the least-loaded key to become available
        min_wait = float("inf")
        best_idx = 0

        for idx in range(self.num_keys):
            if self._daily_credits[idx] >= self._credits_per_key:
                continue  # Truly exhausted

            if self._request_times[idx]:
                oldest = self._request_times[idx][0]
                wait = 60 - (time.monotonic() - oldest)
                if wait < min_wait:
                    min_wait = wait
                    best_idx = idx

        if min_wait < float("inf") and min_wait > 0:
            logger.warning("rate_limit_wait", wait_seconds=round(min_wait, 1), key_index=best_idx)
            await asyncio.sleep(min_wait + 0.1)
            self._request_times[best_idx] = [
                t for t in self._request_times[best_idx]
                if time.monotonic() - t < 60
            ]
            self._request_times[best_idx].append(time.monotonic())
            self._daily_credits[best_idx] += 1
            self._current_index = (best_idx + 1) % self.num_keys
            return self._keys[best_idx]

        raise RuntimeError(
            f"All {self.num_keys} Twelve Data API keys exhausted for today. "
            f"Total credits used: {sum(self._daily_credits.values())}. "
            f"Credits reset at midnight ET."
        )

    def get_stats(self) -> dict[str, Any]:
        """Get key rotation statistics."""
        self._maybe_reset_daily()
        return {
            "num_keys": self.num_keys,
            "total_credits_remaining": self.total_credits_remaining,
            "credits_per_key": {
                f"key_{i+1}": {
                    "used": self._daily_credits[i],
                    "remaining": max(0, self._credits_per_key - self._daily_credits[i]),
                }
                for i in range(self.num_keys)
            },
            "date": self._credits_date.isoformat(),
        }


class TwelveDataAdapter:
    """
    Twelve Data API adapter with multi-key rotation.
    
    Primary use: Server-side technical indicator computation.
    Secondary use: Price data (backup to Alpaca).
    
    Usage:
        td = TwelveDataAdapter()
        await td.connect()
        
        # Get RSI for AAPL
        rsi = await td.get_indicator("AAPL", "rsi", time_period=14)
        
        # Get multiple indicators at once
        data = await td.get_technical_analysis("AAPL", ["rsi", "macd", "bbands"])
        
        # Get OHLCV bars
        bars = await td.get_time_series("AAPL", interval="1day", outputsize=100)
    """

    def __init__(self) -> None:
        self._settings = get_settings().twelve_data
        self._client: httpx.AsyncClient | None = None
        self._rotator: KeyRotator | None = None
        self._request_count = 0

    async def connect(self) -> None:
        """Initialize HTTP client and key rotator."""
        active_keys = self._settings.get_active_keys()

        if not active_keys:
            raise RuntimeError(
                "No Twelve Data API keys configured. "
                "Set TWELVE_DATA_API_KEY (and optionally _2 through _5) in .env"
            )

        self._rotator = KeyRotator(
            keys=active_keys,
            rate_per_key=self._settings.rate_limit_per_key,
            credits_per_key=self._settings.credits_per_key,
        )

        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        logger.info(
            "twelve_data_connected",
            num_keys=len(active_keys),
            total_daily_credits=self._settings.total_daily_credits,
            aggregate_rate=self._settings.aggregate_rate_limit,
        )

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("twelve_data_disconnected")

    # ── Core Request Method ─────────────────

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Make an API request with automatic key rotation.
        
        Handles:
        - Key rotation (round-robin)
        - Rate limiting (per-key)
        - Error handling and retries
        """
        if not self._client or not self._rotator:
            raise RuntimeError("TwelveDataAdapter not connected. Call connect() first.")

        import time
        if time.time() < getattr(self, "_exhausted_until", 0.0):
            raise RuntimeError("Twelve Data daily credits exhausted for today. Skipping to avoid lag.")

        api_key = await self._rotator.get_key()
        params["apikey"] = api_key

        response = await self._client.get(endpoint, params=params)
        self._request_count += 1

        if response.status_code != 200:
            logger.error(
                "twelve_data_http_error",
                endpoint=endpoint,
                status=response.status_code,
                body=response.text[:300],
            )
            if "run out of api credits" in response.text.lower():
                self._exhausted_until = time.time() + 3600.0
                logger.warning("twelve_data_all_credits_exhausted_pausing_1h")
            raise RuntimeError(f"Twelve Data API error: {response.status_code}")

        data = response.json()

        # Check for API-level errors
        if isinstance(data, dict) and data.get("status") == "error":
            error_msg = data.get("message", "Unknown error")
            error_code = data.get("code", 0)

            if error_code == 429:
                # Rate limited — retry with next key
                logger.warning("twelve_data_rate_limited", message=error_msg)
                api_key = await self._rotator.get_key()
                params["apikey"] = api_key
                response = await self._client.get(endpoint, params=params)
                data = response.json()
                if isinstance(data, dict) and data.get("status") == "error":
                    raise RuntimeError(f"Twelve Data API error after retry: {data.get('message')}")
            else:
                raise RuntimeError(f"Twelve Data API error ({error_code}): {error_msg}")

        return data

    # ── Time Series ─────────────────────────

    async def get_time_series(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 100,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get OHLCV time series data.
        
        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            interval: Timeframe ("1min", "5min", "1h", "1day", etc.).
            outputsize: Number of data points (max 5000).
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
        
        Returns:
            List of OHLCV dictionaries.
        """
        td_interval = TD_TIMEFRAME_MAP.get(interval, interval)
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": td_interval,
            "outputsize": min(outputsize, 5000),
            "format": "JSON",
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        data = await self._request("/time_series", params)

        if isinstance(data, dict) and "values" in data:
            return data["values"]
        return []

    # ── Technical Indicators ────────────────

    async def get_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "1day",
        outputsize: int = 100,
        **indicator_params: Any,
    ) -> list[dict[str, Any]]:
        """
        Get a single technical indicator computed server-side.
        
        Args:
            symbol: Ticker symbol.
            indicator: Indicator name (e.g., "rsi", "macd", "bbands").
            interval: Timeframe.
            outputsize: Number of data points.
            **indicator_params: Indicator-specific params (e.g., time_period=14).
        
        Returns:
            List of indicator value dictionaries.
        
        Example:
            rsi = await td.get_indicator("AAPL", "rsi", time_period=14)
            # Returns: [{"datetime": "2026-09-08", "rsi": "65.432"}, ...]
        """
        if indicator.lower() not in AVAILABLE_INDICATORS:
            logger.warning(
                "unknown_indicator",
                indicator=indicator,
                available=len(AVAILABLE_INDICATORS),
            )

        td_interval = TD_TIMEFRAME_MAP.get(interval, interval)
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": td_interval,
            "outputsize": min(outputsize, 5000),
            "format": "JSON",
            **indicator_params,
        }

        data = await self._request(f"/{indicator.lower()}", params)

        if isinstance(data, dict) and "values" in data:
            return data["values"]
        return []

    # ── Batch Indicators ────────────────────

    async def get_technical_analysis(
        self,
        symbol: str,
        indicators: list[str | dict[str, Any]],
        interval: str = "1day",
        outputsize: int = 100,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get multiple indicators for a symbol.
        
        Each indicator costs 1 credit. Use wisely.
        
        Args:
            symbol: Ticker symbol.
            indicators: List of indicator names or dicts with params.
                        e.g., ["rsi", "macd", {"name": "bbands", "time_period": 20}]
            interval: Timeframe.
            outputsize: Number of data points.
        
        Returns:
            Dict mapping indicator name → values list.
        """
        results: dict[str, list[dict[str, Any]]] = {}

        for ind in indicators:
            if isinstance(ind, str):
                name = ind
                params = {}
            else:
                name = ind.get("name", "")
                params = {k: v for k, v in ind.items() if k != "name"}

            try:
                values = await self.get_indicator(
                    symbol=symbol,
                    indicator=name,
                    interval=interval,
                    outputsize=outputsize,
                    **params,
                )
                results[name] = values
                logger.debug(
                    "indicator_fetched",
                    symbol=symbol,
                    indicator=name,
                    data_points=len(values),
                )
            except Exception as e:
                logger.error(
                    "indicator_fetch_failed",
                    symbol=symbol,
                    indicator=name,
                    error=str(e),
                )
                results[name] = []

        return results

    # ── Quote ───────────────────────────────

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote for a symbol."""
        data = await self._request("/quote", {"symbol": symbol})
        return data if isinstance(data, dict) else {}

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get quotes for multiple symbols (comma-separated, 1 credit)."""
        data = await self._request(
            "/quote",
            {"symbol": ",".join(symbols[:20])},  # Max 20 per request
        )

        if isinstance(data, list):
            return {item.get("symbol", ""): item for item in data}
        elif isinstance(data, dict) and "symbol" in data:
            return {data["symbol"]: data}
        return {}

    # ── Exchange Info ────────────────────────

    async def get_stocks_list(
        self,
        exchange: str = "NYSE",
        type: str = "Common Stock",
    ) -> list[dict[str, Any]]:
        """Get list of available stocks (no credit cost)."""
        data = await self._request(
            "/stocks",
            {"exchange": exchange, "type": type},
        )
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return []

    # ── Health & Stats ──────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Check Twelve Data adapter health."""
        if not self._client or not self._rotator:
            return {"status": "disconnected"}

        try:
            # Lightweight check — get API usage (no credit cost)
            api_key = self._rotator._keys[0]
            response = await self._client.get(
                "/api_usage",
                params={"apikey": api_key},
            )

            rotator_stats = self._rotator.get_stats()

            return {
                "status": "healthy",
                "provider": "twelve_data",
                "request_count": self._request_count,
                "key_rotation": rotator_stats,
                "api_usage": response.json() if response.status_code == 200 else None,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            "request_count": self._request_count,
            "key_rotation": self._rotator.get_stats() if self._rotator else None,
        }
