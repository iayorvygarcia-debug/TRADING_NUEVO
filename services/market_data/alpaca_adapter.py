"""
AI Quant Trading Platform — Alpaca Market Data Adapter

Implements MarketDataProvider using Alpaca Markets API.

Data sources:
- REST API: Historical bars, snapshots, latest quotes
- WebSocket: Real-time bars, quotes, trades (event-driven)

Feeds:
- IEX (free): Real-time data from IEX exchange only
- SIP ($99/mo): Full consolidated feed from all exchanges

Documentation: https://docs.alpaca.markets/docs/market-data-api

Key limits (verified Sep 2026):
- Free plan: 200 requests/min, 30 WebSocket symbols
- Algo Trader Plus ($99/mo): 10,000 requests/min, unlimited WS symbols
- Historical: 7+ years of data available
- WebSocket: wss://stream.data.alpaca.markets/v2/{feed}

This adapter handles:
- WebSocket connection with automatic reconnection
- Rate limiting for REST calls
- Data normalization to our Bar/Quote/Trade models
- Health monitoring (latency, connection state)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import httpx

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import DataFeedDisconnectedError, MissingDataError
from services.market_data.base import (
    MarketDataProvider,
    BarCallback,
    QuoteCallback,
    TradeCallback,
)
from services.market_data.models import (
    Bar,
    Quote,
    Trade,
    MarketSnapshot,
    Timeframe,
    QualityStatus,
)

logger = get_logger("alpaca_adapter")

# ── Timeframe mapping: our enum → Alpaca API string ──
TIMEFRAME_MAP = {
    Timeframe.M1: "1Min",
    Timeframe.M5: "5Min",
    Timeframe.M15: "15Min",
    Timeframe.M30: "30Min",
    Timeframe.H1: "1Hour",
    Timeframe.H4: "4Hour",
    Timeframe.D1: "1Day",
    Timeframe.W1: "1Week",
    Timeframe.MN: "1Month",
}


class AlpacaAdapter(MarketDataProvider):
    """
    Alpaca Markets data adapter.
    
    Provides both REST (historical) and WebSocket (real-time) data access.
    Uses httpx for async HTTP and native websockets for streaming.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.alpaca.api_key
        self._secret_key = settings.alpaca.secret_key
        self._data_url = settings.alpaca.data_url
        self._stream_url = settings.alpaca.stream_url
        self._feed = settings.alpaca.feed

        # HTTP client for REST API
        self._client: httpx.AsyncClient | None = None

        # WebSocket state
        self._ws = None
        self._ws_task: asyncio.Task | None = None
        self._ws_connected = False
        self._ws_subscriptions: dict[str, set[str]] = {
            "bars": set(),
            "quotes": set(),
            "trades": set(),
        }

        # Callbacks for real-time data
        self._bar_callbacks: list[BarCallback] = []
        self._quote_callbacks: list[QuoteCallback] = []
        self._trade_callbacks: list[TradeCallback] = []

        # Health metrics
        self._last_message_time: float = 0
        self._message_count: int = 0
        self._error_count: int = 0
        self._connect_time: float = 0

        # Rate limiting
        self._request_timestamps: list[float] = []
        self._max_requests_per_minute = 200  # Free tier default

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def supports_websocket(self) -> bool:
        return True

    # ── HTTP Headers ────────────────────────

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept": "application/json",
        }

    # ── Rate Limiter ────────────────────────

    async def _rate_limit(self) -> None:
        """Enforce rate limiting for REST API calls."""
        now = time.monotonic()
        # Remove timestamps older than 60 seconds
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < 60
        ]
        if len(self._request_timestamps) >= self._max_requests_per_minute:
            wait_time = 60 - (now - self._request_timestamps[0])
            if wait_time > 0:
                logger.warning("rate_limit_wait", wait_seconds=round(wait_time, 2))
                await asyncio.sleep(wait_time)
        self._request_timestamps.append(time.monotonic())

    # ── Lifecycle ───────────────────────────

    async def connect(self) -> None:
        """Initialize HTTP client and validate API keys."""
        if not self._api_key or not self._secret_key:
            raise DataFeedDisconnectedError(
                "Alpaca API keys not configured. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in .env file.",
                details={"provider": "alpaca"},
            )

        self._client = httpx.AsyncClient(
            base_url=self._data_url,
            headers=self._headers,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        # Validate connection by requesting a known symbol snapshot
        try:
            response = await self._client.get(
                f"/v2/stocks/AAPL/snapshot",
                params={"feed": self._feed},
            )
            if response.status_code == 200:
                self._connect_time = time.monotonic()
                logger.info(
                    "alpaca_connected",
                    feed=self._feed,
                    data_url=self._data_url,
                )
            elif response.status_code == 403:
                raise DataFeedDisconnectedError(
                    "Alpaca API keys are invalid or unauthorized.",
                    details={"status": 403, "response": response.text},
                )
            else:
                logger.warning(
                    "alpaca_connect_warning",
                    status=response.status_code,
                    body=response.text[:200],
                )
        except httpx.ConnectError as e:
            raise DataFeedDisconnectedError(
                f"Cannot connect to Alpaca data API: {e}",
                details={"url": self._data_url},
            )

    async def disconnect(self) -> None:
        """Close HTTP client and WebSocket connections."""
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None
            self._ws_connected = False

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("alpaca_disconnected")

    # ── REST: Historical Bars ───────────────

    async def get_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Bar]:
        """
        Fetch historical bars from Alpaca REST API.
        
        Alpaca endpoint: GET /v2/stocks/{symbol}/bars
        
        Handles pagination automatically (Alpaca returns max 10,000 per request).
        """
        if self._client is None:
            raise DataFeedDisconnectedError("Alpaca client not connected")

        alpaca_tf = TIMEFRAME_MAP.get(timeframe)
        if not alpaca_tf:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        all_bars: list[Bar] = []
        page_token: str | None = None
        max_per_page = min(limit or 10_000, 10_000)

        while True:
            await self._rate_limit()

            params: dict[str, Any] = {
                "timeframe": alpaca_tf,
                "start": start.isoformat(),
                "feed": self._feed,
                "limit": max_per_page,
                "adjustment": "all",  # Include splits/dividends adjustments
                "sort": "asc",
            }

            if end:
                params["end"] = end.isoformat()
            if page_token:
                params["page_token"] = page_token

            response = await self._client.get(
                f"/v2/stocks/{symbol}/bars",
                params=params,
            )

            if response.status_code != 200:
                logger.error(
                    "alpaca_bars_error",
                    symbol=symbol,
                    status=response.status_code,
                    body=response.text[:300],
                )
                break

            data = response.json()
            raw_bars = data.get("bars", [])

            if not raw_bars:
                break

            for b in raw_bars:
                bar = Bar(
                    symbol=symbol,
                    timeframe=timeframe,
                    time=datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                    open=float(b["o"]),
                    high=float(b["h"]),
                    low=float(b["l"]),
                    close=float(b["c"]),
                    volume=int(b["v"]),
                    vwap=float(b["vw"]) if "vw" in b else None,
                    trade_count=b.get("n"),
                    source="alpaca",
                    quality_status=QualityStatus.VALID,
                )
                all_bars.append(bar)

            # Check for pagination
            page_token = data.get("next_page_token")
            if not page_token:
                break

            # Respect limit if specified
            if limit and len(all_bars) >= limit:
                all_bars = all_bars[:limit]
                break

        logger.info(
            "alpaca_bars_fetched",
            symbol=symbol,
            timeframe=timeframe.value,
            count=len(all_bars),
            start=start.isoformat(),
            end=end.isoformat() if end else "now",
        )

        return all_bars

    async def get_bars_stream(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime | None = None,
    ) -> AsyncGenerator[Bar, None]:
        """
        Stream historical bars one at a time.
        
        Memory-efficient for large datasets (backtesting, backfill).
        """
        if self._client is None:
            raise DataFeedDisconnectedError("Alpaca client not connected")

        alpaca_tf = TIMEFRAME_MAP.get(timeframe)
        if not alpaca_tf:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        page_token: str | None = None

        while True:
            await self._rate_limit()

            params: dict[str, Any] = {
                "timeframe": alpaca_tf,
                "start": start.isoformat(),
                "feed": self._feed,
                "limit": 10_000,
                "adjustment": "all",
                "sort": "asc",
            }
            if end:
                params["end"] = end.isoformat()
            if page_token:
                params["page_token"] = page_token

            response = await self._client.get(
                f"/v2/stocks/{symbol}/bars",
                params=params,
            )

            if response.status_code != 200:
                logger.error(
                    "alpaca_bars_stream_error",
                    symbol=symbol,
                    status=response.status_code,
                )
                break

            data = response.json()
            raw_bars = data.get("bars", [])

            if not raw_bars:
                break

            for b in raw_bars:
                yield Bar(
                    symbol=symbol,
                    timeframe=timeframe,
                    time=datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                    open=float(b["o"]),
                    high=float(b["h"]),
                    low=float(b["l"]),
                    close=float(b["c"]),
                    volume=int(b["v"]),
                    vwap=float(b["vw"]) if "vw" in b else None,
                    trade_count=b.get("n"),
                    source="alpaca",
                )

            page_token = data.get("next_page_token")
            if not page_token:
                break

    # ── REST: Multi-symbol Bars ─────────────

    async def get_multi_bars(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        start: datetime,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> dict[str, list[Bar]]:
        """
        Fetch bars for multiple symbols in one request.
        
        Alpaca endpoint: GET /v2/stocks/bars
        More efficient than individual requests for universe scans.
        """
        if self._client is None:
            raise DataFeedDisconnectedError("Alpaca client not connected")

        alpaca_tf = TIMEFRAME_MAP.get(timeframe)
        if not alpaca_tf:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        await self._rate_limit()

        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "timeframe": alpaca_tf,
            "start": start.isoformat(),
            "feed": self._feed,
            "limit": min(limit, 10_000),
            "adjustment": "all",
            "sort": "asc",
        }
        if end:
            params["end"] = end.isoformat()

        response = await self._client.get("/v2/stocks/bars", params=params)

        if response.status_code != 200:
            logger.error(
                "alpaca_multi_bars_error",
                status=response.status_code,
                body=response.text[:300],
            )
            return {}

        data = response.json()
        result: dict[str, list[Bar]] = {}

        bars_data = data.get("bars", {})
        for sym, raw_bars in bars_data.items():
            result[sym] = [
                Bar(
                    symbol=sym,
                    timeframe=timeframe,
                    time=datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                    open=float(b["o"]),
                    high=float(b["h"]),
                    low=float(b["l"]),
                    close=float(b["c"]),
                    volume=int(b["v"]),
                    vwap=float(b["vw"]) if "vw" in b else None,
                    trade_count=b.get("n"),
                    source="alpaca",
                )
                for b in raw_bars
            ]

        logger.info(
            "alpaca_multi_bars_fetched",
            symbols_count=len(symbols),
            total_bars=sum(len(v) for v in result.values()),
        )

        return result

    # ── REST: Snapshots ─────────────────────

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """Get current market snapshot for a symbol."""
        if self._client is None:
            raise DataFeedDisconnectedError("Alpaca client not connected")

        await self._rate_limit()

        response = await self._client.get(
            f"/v2/stocks/{symbol}/snapshot",
            params={"feed": self._feed},
        )

        if response.status_code != 200:
            raise MissingDataError(
                f"Cannot get snapshot for {symbol}",
                details={"status": response.status_code},
            )

        data = response.json()
        latest_trade = data.get("latestTrade", {})
        latest_quote = data.get("latestQuote", {})
        daily_bar = data.get("dailyBar", {})
        prev_daily_bar = data.get("prevDailyBar", {})

        last_price = float(latest_trade.get("p", 0))
        prev_close = float(prev_daily_bar.get("c", 0))

        return MarketSnapshot(
            symbol=symbol,
            time=datetime.now(timezone.utc),
            last_price=last_price,
            bid=float(latest_quote.get("bp", 0)),
            ask=float(latest_quote.get("ap", 0)),
            spread=float(latest_quote.get("ap", 0)) - float(latest_quote.get("bp", 0)),
            volume=int(daily_bar.get("v", 0)),
            vwap=float(daily_bar.get("vw", 0)) if "vw" in daily_bar else None,
            daily_change=last_price - prev_close if prev_close else None,
            daily_change_pct=(
                ((last_price - prev_close) / prev_close * 100)
                if prev_close else None
            ),
        )

    async def get_snapshots(self, symbols: list[str]) -> dict[str, MarketSnapshot]:
        """Get snapshots for multiple symbols."""
        if self._client is None:
            raise DataFeedDisconnectedError("Alpaca client not connected")

        await self._rate_limit()

        response = await self._client.get(
            "/v2/stocks/snapshots",
            params={"symbols": ",".join(symbols), "feed": self._feed},
        )

        if response.status_code != 200:
            logger.error("alpaca_snapshots_error", status=response.status_code)
            return {}

        data = response.json()
        result: dict[str, MarketSnapshot] = {}

        for sym, snap in data.items():
            latest_trade = snap.get("latestTrade", {})
            latest_quote = snap.get("latestQuote", {})
            daily_bar = snap.get("dailyBar", {})
            prev_daily_bar = snap.get("prevDailyBar", {})

            last_price = float(latest_trade.get("p", 0))
            prev_close = float(prev_daily_bar.get("c", 0))

            result[sym] = MarketSnapshot(
                symbol=sym,
                time=datetime.now(timezone.utc),
                last_price=last_price,
                bid=float(latest_quote.get("bp", 0)),
                ask=float(latest_quote.get("ap", 0)),
                spread=float(latest_quote.get("ap", 0)) - float(latest_quote.get("bp", 0)),
                volume=int(daily_bar.get("v", 0)),
                vwap=float(daily_bar.get("vw", 0)) if "vw" in daily_bar else None,
                daily_change=last_price - prev_close if prev_close else None,
                daily_change_pct=(
                    ((last_price - prev_close) / prev_close * 100)
                    if prev_close else None
                ),
            )

        return result

    # ── REST: Latest Quote ──────────────────

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        """Get latest bid/ask quote."""
        if self._client is None:
            raise DataFeedDisconnectedError("Alpaca client not connected")

        await self._rate_limit()

        response = await self._client.get(
            f"/v2/stocks/{symbol}/quotes/latest",
            params={"feed": self._feed},
        )

        if response.status_code != 200:
            return None

        data = response.json()
        q = data.get("quote", {})

        return Quote(
            symbol=symbol,
            time=datetime.fromisoformat(q["t"].replace("Z", "+00:00")),
            bid=float(q.get("bp", 0)),
            ask=float(q.get("ap", 0)),
            bid_size=int(q.get("bs", 0)),
            ask_size=int(q.get("as", 0)),
            source="alpaca",
        )

    # ── WebSocket: Real-time Streaming ──────

    async def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        callback: BarCallback,
    ) -> None:
        """Subscribe to real-time bar updates via WebSocket."""
        self._bar_callbacks.append(callback)
        self._ws_subscriptions["bars"].update(symbols)
        await self._ensure_ws_connected()
        await self._send_ws_subscribe()
        logger.info("ws_bars_subscribed", symbols=symbols, timeframe=timeframe.value)

    async def subscribe_quotes(
        self,
        symbols: list[str],
        callback: QuoteCallback,
    ) -> None:
        """Subscribe to real-time quote updates."""
        self._quote_callbacks.append(callback)
        self._ws_subscriptions["quotes"].update(symbols)
        await self._ensure_ws_connected()
        await self._send_ws_subscribe()
        logger.info("ws_quotes_subscribed", symbols=symbols)

    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: TradeCallback,
    ) -> None:
        """Subscribe to real-time trade updates."""
        self._trade_callbacks.append(callback)
        self._ws_subscriptions["trades"].update(symbols)
        await self._ensure_ws_connected()
        await self._send_ws_subscribe()
        logger.info("ws_trades_subscribed", symbols=symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        for channel in self._ws_subscriptions:
            self._ws_subscriptions[channel] -= set(symbols)
        logger.info("ws_unsubscribed", symbols=symbols)

    async def _ensure_ws_connected(self) -> None:
        """Ensure WebSocket is connected, start if not."""
        if not self._ws_connected:
            self._ws_task = asyncio.create_task(self._ws_run_loop())
            # Wait briefly for connection to establish
            for _ in range(50):  # 5 seconds max
                if self._ws_connected:
                    break
                await asyncio.sleep(0.1)

    async def _ws_run_loop(self) -> None:
        """
        WebSocket main loop with automatic reconnection.
        
        Implements exponential backoff on connection failures.
        Max reconnect delay: 60 seconds.
        """
        import json
        import websockets

        reconnect_delay = 1.0
        max_delay = 60.0
        ws_url = f"{self._stream_url}/{self._feed}"

        while True:
            try:
                logger.info("ws_connecting", url=ws_url)

                async with websockets.connect(
                    ws_url,
                    additional_headers={
                        "APCA-API-KEY-ID": self._api_key,
                        "APCA-API-SECRET-KEY": self._secret_key,
                    },
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._ws_connected = True
                    reconnect_delay = 1.0  # Reset on successful connect
                    logger.info("ws_connected", feed=self._feed)

                    # Authenticate
                    auth_msg = json.dumps({
                        "action": "auth",
                        "key": self._api_key,
                        "secret": self._secret_key,
                    })
                    await ws.send(auth_msg)

                    # Wait for auth response
                    auth_response = await ws.recv()
                    auth_data = json.loads(auth_response)
                    logger.info("ws_auth_response", data=str(auth_data)[:200])

                    # Subscribe to channels
                    await self._send_ws_subscribe()

                    # Process messages
                    async for message in ws:
                        await self._handle_ws_message(message)

            except asyncio.CancelledError:
                logger.info("ws_loop_cancelled")
                break
            except Exception as e:
                self._ws_connected = False
                self._error_count += 1
                logger.error(
                    "ws_error",
                    error=str(e),
                    reconnect_delay=reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)

    async def _send_ws_subscribe(self) -> None:
        """Send subscription message to WebSocket."""
        import json

        if not self._ws or not self._ws_connected:
            return

        sub_msg = {
            "action": "subscribe",
            "bars": list(self._ws_subscriptions["bars"]),
            "quotes": list(self._ws_subscriptions["quotes"]),
            "trades": list(self._ws_subscriptions["trades"]),
        }

        await self._ws.send(json.dumps(sub_msg))
        logger.debug("ws_subscribe_sent", subscriptions=sub_msg)

    async def _handle_ws_message(self, raw_message: str) -> None:
        """
        Parse and dispatch WebSocket messages.
        
        Alpaca sends messages as JSON arrays: [{"T": "b", ...}, ...]
        T = message type: "b" (bar), "q" (quote), "t" (trade)
        """
        import json

        self._last_message_time = time.monotonic()
        self._message_count += 1

        try:
            messages = json.loads(raw_message)
            if not isinstance(messages, list):
                messages = [messages]

            for msg in messages:
                msg_type = msg.get("T")

                if msg_type == "b":
                    # Bar update
                    bar = Bar(
                        symbol=msg["S"],
                        timeframe=Timeframe.M1,  # Alpaca WS sends minute bars
                        time=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                        open=float(msg["o"]),
                        high=float(msg["h"]),
                        low=float(msg["l"]),
                        close=float(msg["c"]),
                        volume=int(msg["v"]),
                        vwap=float(msg["vw"]) if "vw" in msg else None,
                        trade_count=msg.get("n"),
                        source="alpaca_ws",
                    )
                    for callback in self._bar_callbacks:
                        try:
                            await callback(bar)
                        except Exception as e:
                            logger.error("bar_callback_error", error=str(e))

                elif msg_type == "q":
                    # Quote update
                    quote = Quote(
                        symbol=msg["S"],
                        time=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                        bid=float(msg.get("bp", 0)),
                        ask=float(msg.get("ap", 0)),
                        bid_size=int(msg.get("bs", 0)),
                        ask_size=int(msg.get("as", 0)),
                        source="alpaca_ws",
                    )
                    for callback in self._quote_callbacks:
                        try:
                            await callback(quote)
                        except Exception as e:
                            logger.error("quote_callback_error", error=str(e))

                elif msg_type == "t":
                    # Trade update
                    trade = Trade(
                        symbol=msg["S"],
                        time=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                        price=float(msg["p"]),
                        size=int(msg["s"]),
                        exchange=msg.get("x", ""),
                        conditions=",".join(msg.get("c", [])),
                        source="alpaca_ws",
                    )
                    for callback in self._trade_callbacks:
                        try:
                            await callback(trade)
                        except Exception as e:
                            logger.error("trade_callback_error", error=str(e))

                elif msg_type in ("success", "subscription"):
                    logger.debug("ws_control_message", type=msg_type, msg=msg)

                elif msg_type == "error":
                    logger.error("ws_error_message", msg=msg)
                    self._error_count += 1

        except json.JSONDecodeError as e:
            logger.error("ws_json_error", error=str(e), raw=raw_message[:200])
        except Exception as e:
            logger.error("ws_handle_error", error=str(e))
            self._error_count += 1

    # ── Health Check ────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Check Alpaca adapter health."""
        now = time.monotonic()
        rest_healthy = self._client is not None

        # Check REST with a lightweight call
        rest_latency_ms: float | None = None
        if self._client:
            try:
                start = time.monotonic()
                response = await self._client.get(
                    "/v2/stocks/AAPL/quotes/latest",
                    params={"feed": self._feed},
                )
                rest_latency_ms = round((time.monotonic() - start) * 1000, 2)
                rest_healthy = response.status_code == 200
            except Exception:
                rest_healthy = False

        # WebSocket health
        ws_stale = False
        if self._ws_connected and self._last_message_time > 0:
            time_since_last = now - self._last_message_time
            ws_stale = time_since_last > 60  # No message for 60s = stale

        status = "healthy"
        if not rest_healthy:
            status = "unhealthy"
        elif ws_stale:
            status = "degraded"

        return {
            "status": status,
            "provider": "alpaca",
            "feed": self._feed,
            "rest": {
                "connected": rest_healthy,
                "latency_ms": rest_latency_ms,
            },
            "websocket": {
                "connected": self._ws_connected,
                "stale": ws_stale,
                "subscriptions": {
                    k: len(v) for k, v in self._ws_subscriptions.items()
                },
                "message_count": self._message_count,
                "error_count": self._error_count,
            },
        }
