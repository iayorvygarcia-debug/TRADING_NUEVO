"""
AI Quant Trading Platform — FastAPI Gateway

Main entry point for the platform's REST API.
Handles:
- Health checks for all services
- System status and dashboard data
- Configuration management
- Manual controls (kill switch, trading mode)

This is NOT where trading logic lives.
Trading logic lives in services/ and strategies/.
The API only exposes controls and monitoring.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.logging import setup_logging, get_logger
from core.database import init_database, close_database
from core.redis_client import init_redis, close_redis
from core.market_session import get_session_info
from core.security import validate_api_keys
from core import database as db_module
from core import redis_client as redis_module
from core import event_bus
from apps.worker.quant_trading_worker import get_global_worker

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle: startup and shutdown."""
    settings = get_settings()

    # ── Startup ─────────────────────────────
    setup_logging()
    logger.info(
        "platform_starting",
        environment=settings.environment.value,
        trading_mode=settings.trading_mode.value,
    )

    # Initialize infrastructure
    try:
        await init_database()
    except Exception as exc:
        logger.warning("database_init_deferred", error=str(exc))

    try:
        await init_redis()
    except Exception as exc:
        logger.warning("redis_init_deferred", error=str(exc))

    # Validate API keys
    api_status = validate_api_keys()
    logger.info("api_keys_validated", status=api_status)

    logger.info(
        "platform_ready",
        trading_mode=settings.trading_mode.value,
        session=get_session_info(),
    )

    # Launch continuous autonomous quant market scanning loop (every 60s)
    worker = get_global_worker()
    worker_task = asyncio.create_task(worker.start())
    logger.info("quant_worker_background_loop_spawned", interval=worker.scan_interval_seconds)

    print("\n" + "="*60)
    print("  [+] AI QUANT TRADING PLATFORM — INICIADO EXITOSAMENTE")
    print("  [+] Dashboard Web: http://localhost:8000/dashboard")
    print("  [+] API Docs:      http://localhost:8000/docs")
    print("="*60 + "\n", flush=True)

    yield

    # ── Shutdown ────────────────────────────
    logger.info("platform_shutting_down")
    worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    await close_redis()
    await close_database()
    logger.info("platform_stopped")


# ── Create App ──────────────────────────────
app = FastAPI(
    title="AI Quant Trading Platform",
    description="Professional algorithmic trading system API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permite que el dashboard se cargue desde cualquier origen (proxy local, produccion, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & Status Endpoints ───────────────

@app.get("/health", tags=["System"])
async def health_check(deep: bool = False):
    """
    Lightweight health check for Watchdog and uptime monitoring.
    Returns immediately (<5ms). Set ?deep=true for full DB/Redis diagnostic probes.
    """
    session_info = get_session_info()
    settings = get_settings()

    if not deep:
        return {
            "status": "healthy",
            "trading_mode": settings.trading_mode.value,
            "environment": settings.environment.value,
            "market_session": session_info,
        }

    # Deep diagnostic check (only if explicitly requested via ?deep=true)
    import asyncio
    try:
        db_health = await asyncio.wait_for(db_module.health_check(), timeout=0.3)
    except Exception:
        db_health = {"status": "unhealthy"}

    try:
        redis_health = await asyncio.wait_for(redis_module.health_check(), timeout=0.3)
    except Exception:
        redis_health = {"status": "unhealthy"}

    try:
        event_bus_health = await asyncio.wait_for(event_bus.event_bus_health(), timeout=0.3)
    except Exception:
        event_bus_health = {"status": "degraded"}

    services = [db_health, redis_health, event_bus_health]
    overall = "unhealthy" if any(s.get("status") == "unhealthy" for s in services) else ("degraded" if any(s.get("status") == "degraded" for s in services) else "healthy")

    return {
        "status": overall,
        "trading_mode": settings.trading_mode.value,
        "environment": settings.environment.value,
        "market_session": session_info,
        "services": {
            "database": db_health,
            "redis": redis_health,
            "event_bus": event_bus_health,
        },
    }


@app.get("/status", tags=["System"])
async def system_status():
    """
    High-level system status for the dashboard.
    
    Returns trading mode, market session, and summary metrics.
    """
    settings = get_settings()
    session_info = get_session_info()

    return {
        "platform": "AI Quant Trading Platform",
        "version": "0.1.0",
        "trading_mode": settings.trading_mode.value,
        "is_live": settings.is_live,
        "market_session": session_info,
        "risk_config": {
            "risk_per_trade": settings.risk.risk_per_trade,
            "max_daily_loss": settings.risk.max_daily_loss,
            "max_drawdown": settings.risk.max_drawdown,
            "max_position_size": settings.risk.max_position_size,
        },
    }


@app.get("/session", tags=["Market"])
async def market_session():
    """Get current market session information."""
    return get_session_info()


# ── Dashboard & Telemetry Endpoints ─────────
from pathlib import Path
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from apps.worker.quant_trading_worker import get_global_worker
from services.risk_engine.kill_switch import get_kill_switch


DASHBOARD_PATH = Path(__file__).parent / "static" / "dashboard.html"


class KillSwitchRequest(BaseModel):
    reason: str = "Manual API activation"


class ResumeRequest(BaseModel):
    passphrase: str


class ScanRequest(BaseModel):
    auto_trade: bool = False


class AutoTradeRequest(BaseModel):
    enabled: bool


class ShadowModeRequest(BaseModel):
    enabled: bool


class EODModeRequest(BaseModel):
    close_positions_eod: bool



@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/portfolio", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/orders", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/scanner", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/macro", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/sec", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/backtests", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/settings", response_class=HTMLResponse, tags=["Dashboard"])
async def serve_dashboard():
    """Serves the real-time AI Quant Trading OS Web Dashboard."""
    if DASHBOARD_PATH.exists():
        return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Dashboard file not found. Ensure apps/api/static/dashboard.html exists.</h1>",
        status_code=404,
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi import Response
    return Response(status_code=204)


# ── Caché global de telemetría (evita recalcular para cada cliente SSE) ──
_TELEMETRY_SNAPSHOT_CACHE: dict = {"ts": 0.0, "data": None}
_TELEMETRY_LOCK = asyncio.Lock()


async def _build_telemetry_snapshot() -> dict:
    """Construye el snapshot completo con barras y historial. Usa caché de 5s."""
    import time as _time
    now = _time.monotonic()
    async with _TELEMETRY_LOCK:
        if _TELEMETRY_SNAPSHOT_CACHE["data"] is not None and (now - _TELEMETRY_SNAPSHOT_CACHE["ts"]) < 5.0:
            return _TELEMETRY_SNAPSHOT_CACHE["data"]

        worker = get_global_worker()
        snapshot = worker.get_telemetry_snapshot()
        positions = snapshot.get("positions", [])
        for pos in positions:
            sym = pos.get("symbol")
            if sym:
                try:
                    pos["bars"] = await get_bars(sym, timeframe="1Min", limit=30)
                except Exception:
                    pos["bars"] = []
        try:
            snapshot["portfolio_history"] = await get_portfolio_history(period="1D", timeframe="1Min")
        except Exception:
            snapshot["portfolio_history"] = []

        _TELEMETRY_SNAPSHOT_CACHE["ts"] = now
        _TELEMETRY_SNAPSHOT_CACHE["data"] = snapshot
        return snapshot


@app.get("/api/v1/telemetry", tags=["Telemetry"])
async def get_telemetry():
    """Snapshot de telemetría completo para el dashboard. Usa caché interno de 5s."""
    return await _build_telemetry_snapshot()


@app.get("/api/v1/stream", tags=["Telemetry"])
async def stream_telemetry():
    """
    Server-Sent Events (SSE) — UNA sola conexion persistente que envia actualizaciones cada 15s.
    Elimina el polling repetitivo: el browser abre 1 conexion en vez de 1 por cada 15 segundos.
    Esto reduce los hits a LiteSpeed/PHP de ~5/min a CERO adicionales.
    """
    import json as _json
    from fastapi.responses import StreamingResponse

    async def event_generator():
        # Heartbeat de bienvenida inmediato
        try:
            snapshot = await _build_telemetry_snapshot()
            yield f"data: {_json.dumps(snapshot)}\n\n"
        except Exception as exc:
            logger.warning("sse_initial_snapshot_failed", error=str(exc))
            yield f"data: {{\"status\": \"connecting\"}}\n\n"

        while True:
            await asyncio.sleep(15)  # Intervalo de 15s — 1 conexion, 0 polls adicionales
            try:
                snapshot = await _build_telemetry_snapshot()
                yield f"data: {_json.dumps(snapshot)}\n\n"
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("sse_push_failed", error=str(exc))
                yield f": heartbeat\n\n"  # Keep-alive silencioso

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Desactiva buffering en Nginx/LiteSpeed
            "Access-Control-Allow-Origin": "*",
        },
    )



# ── Manual Controls ─────────────────────────

@app.post("/api/v1/control/kill-switch", tags=["Control"])
@app.post("/kill-switch", tags=["Risk"])
async def activate_kill_switch(payload: Optional[KillSwitchRequest] = None):
    """
    🛑 KILL SWITCH — Emergency halt all trading.
    
    When activated:
    1. Blocks all new orders via KillSwitch singleton
    2. Cancels active unexecuted broker orders
    3. Publishes emergency event on event bus
    4. Enters Safe Mode
    """
    reason = payload.reason if payload else "Manual API activation"
    ks = get_kill_switch()
    ks.trigger(reason=reason, operator="api_manual")

    worker = get_global_worker()
    await worker.broker.cancel_all_orders()

    logger.critical(
        "kill_switch_activated",
        reason=reason,
        operator="api_manual",
    )

    try:
        await event_bus.publish_event(
            event_type=event_bus.EventType.KILL_SWITCH_ACTIVATED,
            data={"reason": reason, "operator": "api_manual"},
            source="api",
        )
    except Exception as exc:
        logger.warning("kill_switch_event_publish_failed", error=str(exc))

    _TELEMETRY_SNAPSHOT_CACHE["ts"] = 0.0
    _TELEMETRY_SNAPSHOT_CACHE["data"] = None

    return {
        "status": "kill_switch_activated",
        "reason": reason,
        "is_active": ks.is_active,
        "message": "All new trading halted and working orders cancelled.",
    }


@app.post("/api/v1/control/resume", tags=["Control"])
async def resume_trading(payload: ResumeRequest):
    """
    🔓 Resumes trading by verifying passphrase CONFIRM_RESUME.
    Also resets Circuit Breaker to NORMAL and re-synchronizes daily equity base.
    """
    ks = get_kill_switch()
    resumed = ks.resume(passphrase=payload.passphrase)
    if not resumed:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail="Invalid passphrase. Operation requires 'CONFIRM_RESUME'.",
        )

    from services.risk_engine.circuit_breaker import get_circuit_breaker
    cb = get_circuit_breaker()
    cb.reset()
    worker = get_global_worker()
    curr_eq = getattr(worker.broker, "equity", 100000.0)
    worker.daily_starting_equity = curr_eq
    worker.daily_peak_equity = curr_eq
    worker._save_worker_config()

    _TELEMETRY_SNAPSHOT_CACHE["ts"] = 0.0
    _TELEMETRY_SNAPSHOT_CACHE["data"] = None

    logger.info("kill_switch_and_circuit_breaker_resumed", operator="api_manual")
    return {
        "status": "trading_resumed",
        "kill_switch_active": ks.is_active,
        "circuit_breaker_tier": cb.level.name,
        "message": "Trading successfully resumed and Circuit Breaker reset to NORMAL.",
    }


@app.post("/api/v1/control/circuit-breaker/reset", tags=["Control"])
async def reset_circuit_breaker():
    """
    Desbloquea y reinicia el Circuit Breaker de vuelta a NORMAL,
    actualizando la base diaria de equity para reanudar operaciones inmediatamente.
    """
    from services.risk_engine.circuit_breaker import get_circuit_breaker
    cb = get_circuit_breaker()
    cb.reset()
    worker = get_global_worker()
    curr_eq = getattr(worker.broker, "equity", 100000.0)
    worker.daily_starting_equity = curr_eq
    worker.daily_peak_equity = curr_eq
    worker._save_worker_config()
    _TELEMETRY_SNAPSHOT_CACHE["ts"] = 0.0
    _TELEMETRY_SNAPSHOT_CACHE["data"] = None
    logger.info("circuit_breaker_manually_reset")
    return {
        "status": "circuit_breaker_reset",
        "circuit_breaker_tier": cb.level.name,
        "message": "Circuit Breaker restablecido a NORMAL. Compras desbloqueadas.",
    }


_BARS_CACHE: dict = {}

@app.get("/api/v1/market-data/bars/{symbol}", tags=["Market Data"])
async def get_bars(
    symbol: str,
    timeframe: str = "1Min",
    limit: int = 60,
):
    """
    Fetch recent OHLCV bars for a symbol directly from Alpaca.
    Uses 45s in-memory caching and reverse sorting to always get current market bars.
    """
    import time
    import httpx
    from datetime import datetime, timezone, timedelta

    cache_key = f"{symbol.upper()}:{timeframe}:{limit}"
    now_epoch = time.time()
    if cache_key in _BARS_CACHE:
        cached_ts, cached_data = _BARS_CACHE[cache_key]
        if now_epoch - cached_ts < 45.0:
            return cached_data

    settings = get_settings()
    # Retroceder 5 días para asegurar barras recientes incluso tras fines de semana o pre-market
    start = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()

    headers = {
        "APCA-API-KEY-ID": settings.alpaca.api_key,
        "APCA-API-SECRET-KEY": settings.alpaca.secret_key,
    }

    params = {
        "timeframe": timeframe,
        "start": start,
        "limit": limit,
        "feed": settings.alpaca.feed,
        "adjustment": "all",
        "sort": "desc",
    }

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(
                f"{settings.alpaca.data_url}/v2/stocks/{symbol.upper()}/bars",
                headers=headers,
                params=params,
            )
        if resp.status_code == 200:
            raw = resp.json().get("bars", [])
            if raw:
                # Ordenar cronológicamente ascendente para renderizar velas de izquierda a derecha
                raw.sort(key=lambda b: b.get("t", ""))
                _BARS_CACHE[cache_key] = (now_epoch, raw)
                return raw
    except Exception as exc:
        logger.warning("alpaca_bars_fetch_error", symbol=symbol, error=str(exc))

    # Si Alpaca no devuelve barras para este símbolo, buscar el precio real del worker
    worker = get_global_worker()
    pos = worker.broker._positions.get(symbol.upper()) if hasattr(worker.broker, "_positions") else None
    base_price = (
        float(pos.current_price or pos.avg_entry_price)
        if pos and (pos.current_price or pos.avg_entry_price)
        else 100.0
    )

    now_ts = datetime.now(timezone.utc)
    fallback_bars = [
        {
            "t": (now_ts - timedelta(minutes=i)).isoformat(),
            "o": round(base_price * (1 - 0.001 * (1 if i % 2 == 0 else -1)), 2),
            "h": round(base_price * 1.0015, 2),
            "l": round(base_price * 0.9985, 2),
            "c": round(base_price, 2),
            "v": 1000,
        }
        for i in range(min(limit, 10), -1, -1)
    ]
    return fallback_bars


@app.get("/api/v1/trade-history", tags=["Market Data"])
async def get_trade_history(limit: int = 50):
    """
    Fetch recent closed/filled orders from active broker or Alpaca paper account.
    Shows what the bot actually bought and sold, with P&L per trade.
    """
    worker = get_global_worker()
    paper_trades = worker.broker.get_trade_history()
    if paper_trades:
        return paper_trades[:limit]

    import httpx

    settings = get_settings()
    headers = {
        "APCA-API-KEY-ID": settings.alpaca.api_key,
        "APCA-API-SECRET-KEY": settings.alpaca.secret_key,
    }

    broker_url = settings.alpaca.base_url  # paper-api.alpaca.markets

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                f"{broker_url}/v2/orders",
                headers=headers,
                params={
                    "status": "closed",
                    "limit": limit,
                    "direction": "desc",
                    "nested": "true",
                },
            )

        if resp.status_code == 200:
            orders = resp.json()
            result = []
            for o in orders:
                if o.get("status") not in ("filled", "partially_filled"):
                    continue

                filled_qty  = float(o.get("filled_qty") or 0)
                filled_avg  = float(o.get("filled_avg_price") or 0)
                side        = o.get("side", "")
                symbol      = o.get("symbol", "")
                order_type  = o.get("order_type", o.get("type", ""))
                submitted   = o.get("submitted_at", "")
                filled_at   = o.get("filled_at", "")

                result.append({
                    "order_id":    o.get("id", "")[:12],
                    "symbol":      symbol,
                    "side":        side,
                    "qty":         filled_qty,
                    "avg_price":   filled_avg,
                    "notional":    round(filled_qty * filled_avg, 2),
                    "order_type":  order_type,
                    "status":      o.get("status", ""),
                    "submitted_at": submitted,
                    "filled_at":   filled_at,
                })
            return result

    except Exception as exc:
        logger.warning("alpaca_trade_history_unreachable", error=str(exc))

    return paper_trades[:limit] if paper_trades else []


_PORTFOLIO_CACHE: dict = {}

@app.get("/api/v1/portfolio-history", tags=["Market Data"])
async def get_portfolio_history(period: str = "1D", timeframe: str = "1Min"):
    """
    Fetch portfolio equity history from Alpaca or local broker.
    Used by the equity curve chart for historical data.
    Uses 60s in-memory cache to prevent redundant broker calls.
    """
    import time
    now_epoch = time.time()
    cache_key = f"{period}:{timeframe}"
    if cache_key in _PORTFOLIO_CACHE:
        cached_ts, cached_data = _PORTFOLIO_CACHE[cache_key]
        if now_epoch - cached_ts < 60.0:
            return cached_data

    import httpx
    from datetime import datetime, timezone

    settings = get_settings()
    headers = {
        "APCA-API-KEY-ID": settings.alpaca.api_key,
        "APCA-API-SECRET-KEY": settings.alpaca.secret_key,
    }

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                f"{settings.alpaca.base_url}/v2/account/portfolio/history",
                headers=headers,
                params={"period": period, "timeframe": timeframe, "intraday_reporting": "market_hours"},
            )

        if resp.status_code == 200:
            data = resp.json()
            timestamps = data.get("timestamp", [])
            equity     = data.get("equity", [])
            profit_loss = data.get("profit_loss", [])

            history = [
                {
                    "t": ts,
                    "equity": eq,
                    "pnl": pl,
                }
                for ts, eq, pl in zip(timestamps, equity, profit_loss)
                if eq is not None
            ]
            if history:
                _PORTFOLIO_CACHE[cache_key] = (now_epoch, history)
                return history
    except Exception as exc:
        logger.warning("alpaca_portfolio_history_unreachable", error=str(exc))

    worker = get_global_worker()
    curr_cash = getattr(worker.broker, "cash", 100000.0)
    now_utc_epoch = int(datetime.now(timezone.utc).timestamp())
    fallback = [
        {"t": now_utc_epoch - 3600, "equity": curr_cash, "pnl": 0.0},
        {"t": now_utc_epoch, "equity": curr_cash, "pnl": 0.0},
    ]
    _PORTFOLIO_CACHE[cache_key] = (now_epoch, fallback)
    return fallback


@app.post("/api/v1/control/scan", tags=["Control"])
async def trigger_on_demand_scan(payload: Optional[ScanRequest] = None):
    """
    ⚡ Triggers an immediate universe scan and trade evaluation cycle.
    By default (auto_trade=False), it performs pure market scanning, ML veto filtering, and radar ranking without order entry.
    """
    worker = get_global_worker()
    auto_trade = payload.auto_trade if payload is not None else False
    result = await worker.execute_cycle(auto_trade=auto_trade)
    return {
        "status": "cycle_completed",
        "telemetry": result,
    }


@app.post("/api/v1/control/auto-trade", tags=["Control"])
async def toggle_auto_trade(payload: AutoTradeRequest):
    """
    Toggles autonomous order execution ON or OFF.
    """
    worker = get_global_worker()
    worker.set_auto_trade(payload.enabled)
    logger.info("auto_trade_toggled", enabled=worker.auto_trade)
    return {
        "status": "updated",
        "auto_trade": worker.auto_trade,
    }


@app.post("/api/v1/control/shadow-mode", tags=["Control"])
async def toggle_shadow_mode(payload: ShadowModeRequest):
    """
    Toggles Shadow Mode (simulated order evaluation with telemetry without sending to broker).
    """
    worker = get_global_worker()
    worker.set_shadow_mode(payload.enabled)
    logger.info("shadow_mode_toggled", enabled=worker.shadow_mode)
    return {
        "status": "updated",
        "shadow_mode": worker.shadow_mode,
    }


@app.get("/api/v1/shadow/summary", tags=["Shadow Trading"])
async def get_shadow_summary():
    """Returns current shadow trading metrics, open virtual positions, and trade counts."""
    worker = get_global_worker()
    if not hasattr(worker, "shadow_engine"):
        raise HTTPException(status_code=503, detail="Shadow engine not initialized")
    engine = worker.shadow_engine
    return {
        "shadow_mode": worker.shadow_mode,
        "metrics": engine.metrics.to_dict(),
        "open_positions": [p.to_dict() for p in engine.open_positions.values()],
        "total_closed_trades": len(engine.closed_trades),
    }


@app.get("/api/v1/shadow/decisions", tags=["Shadow Trading"])
async def get_shadow_decisions(limit: int = 50):
    """Returns the most recent forensic trading decisions evaluated in shadow mode."""
    worker = get_global_worker()
    if not hasattr(worker, "shadow_engine"):
        raise HTTPException(status_code=503, detail="Shadow engine not initialized")
    decisions = list(worker.shadow_engine.recent_decisions)[-limit:]
    return [d.to_dict() for d in reversed(decisions)]


@app.get("/api/v1/shadow/reconstruction/{symbol}", tags=["Shadow Trading"])
async def get_forensic_reconstruction(symbol: str):
    """
    Returns human-readable forensic audit reconstruction for a given symbol:
    '¿Por qué quiso comprar AAPL?'
    """
    worker = get_global_worker()
    if not hasattr(worker, "shadow_engine"):
        raise HTTPException(status_code=503, detail="Shadow engine not initialized")
    reconstruction = worker.shadow_engine.get_forensic_reconstruction(symbol.upper())
    if not reconstruction:
        raise HTTPException(status_code=404, detail=f"No forensic decision found for symbol {symbol.upper()}")
    return {
        "symbol": symbol.upper(),
        "reconstruction": reconstruction,
    }


@app.get("/api/v1/shadow/report", tags=["Shadow Trading"])
async def get_shadow_report(date: Optional[str] = None):
    """Generates and returns the daily shadow trading report (JSON and Markdown)."""
    worker = get_global_worker()
    if not hasattr(worker, "shadow_engine"):
        raise HTTPException(status_code=503, detail="Shadow engine not initialized")
    report_data, markdown_content = worker.shadow_engine.generate_daily_report(date_str=date)
    return {
        "report": report_data,
        "markdown": markdown_content,
    }



@app.post("/api/v1/control/swing-mode", tags=["Control"])
async def toggle_swing_mode(payload: EODModeRequest):
    """
    Toggles whether positions are kept open overnight (Swing Trading, close_positions_eod=False)
    or forcibly liquidated at 15:55 ET (Day Trading, close_positions_eod=True).
    """
    worker = get_global_worker()
    worker.set_close_positions_eod(payload.close_positions_eod)
    return {
        "status": "updated",
        "close_positions_eod": worker.close_positions_eod,
        "swing_mode": not worker.close_positions_eod,
    }


@app.post("/api/v1/control/reset", tags=["Control"])
async def reset_paper_account():
    """
    Resets the paper trading account back to initial clean state ($1,000.00).
    Clears all positions, orders, and trade history.
    """
    worker = get_global_worker()
    initial_cap = float(getattr(worker.settings.risk, "initial_capital", 1000.0))
    if hasattr(worker.broker, "reset_account"):
        worker.broker.reset_account(initial_cash=initial_cap)
    elif hasattr(worker.broker, "client"):
        try:
            worker.broker.client.cancel_orders()
            worker.broker.client.close_all_positions(cancel_orders=True)
            worker.broker._positions.clear()
            worker.broker._orders.clear()
        except Exception as e:
            logger.warning("reset_positions_error", error=str(e))
    worker.daily_starting_equity = initial_cap
    worker.daily_peak_equity = initial_cap
    from services.risk_engine.circuit_breaker import get_circuit_breaker
    get_circuit_breaker().reset()
    worker.position_entry_times.clear()
    worker.position_be_activated.clear()
    worker.position_peaks.clear()
    worker._save_worker_config()
    _TELEMETRY_SNAPSHOT_CACHE["ts"] = 0.0
    _TELEMETRY_SNAPSHOT_CACHE["data"] = None
    logger.info("paper_account_reset_via_api", initial_cash=initial_cap)
    return {
        "status": "account_reset",
        "message": f"Paper account reset to ${initial_cap:,.2f} with clean history.",
        "telemetry": worker.get_telemetry_snapshot(),
    }


# ── Harness² (Recursive Agent Harnessing) Endpoints ───────────────
@app.get("/api/v1/harness/status", tags=["Harness² AI"])
async def get_harness_status():
    """Returns the executive summary of the Harness² recursive self-tuning engine."""
    from services.harness_engine import get_harness_engine
    return get_harness_engine().get_summary()


@app.get("/api/v1/harness/diagnoses", tags=["Harness² AI"])
async def get_harness_diagnoses(limit: int = 50):
    """Returns recent forensic trade diagnoses explaining causes of losses and wins."""
    from services.harness_engine import get_harness_engine
    engine = get_harness_engine()
    return [d.model_dump() for d in reversed(engine.state.diagnoses[-limit:])]


@app.get("/api/v1/harness/lessons", tags=["Harness² AI"])
async def get_harness_lessons():
    """Returns active evidence-backed rules and lessons derived from historical outcomes."""
    from services.harness_engine import get_harness_engine
    engine = get_harness_engine()
    return [l.model_dump() for l in engine.state.lessons if l.is_active]


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">⚡</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")

