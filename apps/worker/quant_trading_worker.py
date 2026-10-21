"""AI Quant Trading Platform — 24/7 Autonomous Daemon Worker.

Orchestrates the platform's four planes:
1. Data Plane: Real-time bars, OHLCV validation, data quality checks.
2. Intelligence Plane: Multi-timeframe indicators, news sentiment, macro regime, ML inference, universe scanning.
3. Trading Plane: Strategy consensus, deterministic risk gatekeeper, bracket order execution.
4. Control Plane: Kill switch, circuit breaker, reconciliation, telemetry state.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Bootstrap project root into sys.path to allow standalone CLI execution
_root_path = str(Path(__file__).resolve().parent.parent.parent)
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

from core.logging import get_logger
from core.config import get_settings
from core.market_session import (
    MarketSession,
    SessionCapabilities,
    get_current_session,
    get_session_capabilities,
    get_session_info,
    get_dynamic_max_spread,
    ET,
)
from services.broker_engine.base import BrokerInterface
from services.broker_engine.execution_engine import ExecutionEngine
from services.broker_engine.paper_broker import PaperBroker
from services.broker_engine.alpaca_broker import AlpacaBroker
from services.macro_sec.regime_manager import MarketRegimeManager
from services.ml_pipeline.governance import ModelRegistry
from services.ml_pipeline.predictor import MLSignalPredictor
from services.news_engine.impact import NewsImpactEngine
from services.news_engine.alpaca_news import AlpacaNewsAdapter
from services.opportunity_engine.ranker import UniverseScanner
from services.opportunity_engine.universe import UniverseManager
from services.risk_engine.circuit_breaker import get_circuit_breaker
from services.risk_engine.models import RiskLevel
from services.risk_engine.engine import RiskEngine
from services.risk_engine.kill_switch import get_kill_switch
from services.risk_engine.pre_trade_guard import PreTradeRiskGuard
from services.strategy_engine.signal_combiner import SignalCombiner
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.multi_timeframe_confluence import MultiTimeframeConfluenceStrategy
from strategies.committee_agents import build_committee_agents
from services.shadow_engine import ShadowTradingEngine
from services.harness_engine import get_harness_engine


logger = get_logger("quant_trading_worker")


class QuantTradingWorker:
    """Autonomous trading engine coordinating scanning, analysis, risk, and execution."""

    def __init__(
        self,
        broker: Optional[BrokerInterface] = None,
        risk_engine: Optional[RiskEngine] = None,
        regime_manager: Optional[MarketRegimeManager] = None,
        news_engine: Optional[NewsImpactEngine] = None,
        model_registry: Optional[ModelRegistry] = None,
        scan_interval_seconds: float = 60.0,
    ):
        self.settings = get_settings()
        if broker is not None:
            self.broker = broker
        elif self.settings.alpaca.validate_keys():
            logger.info("quant_worker_using_alpaca_official", is_paper=self.settings.alpaca.is_paper)
            self.broker = AlpacaBroker()
        else:
            self.broker = PaperBroker(
                initial_cash=self.settings.risk.initial_capital,
                state_file=Path("data/paper_account_state.json"),
            )
        self.risk_engine = risk_engine or RiskEngine()
        self.execution_engine = ExecutionEngine(broker=self.broker, risk_engine=self.risk_engine)
        self.regime_manager = regime_manager or MarketRegimeManager()
        self.news_engine = news_engine or NewsImpactEngine()
        self.news_adapter = AlpacaNewsAdapter() if self.settings.alpaca.validate_keys() else None
        self.latest_news_scores: Dict[str, float] = {}
        self.model_registry = model_registry or ModelRegistry()
        self.ml_predictor = MLSignalPredictor(registry=self.model_registry)

        # Strategies
        existing_strategies = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            VWAPReversionStrategy(),
            MultiTimeframeConfluenceStrategy(),
        ]
        self.committee_agents = build_committee_agents(existing_strategies)
        self.signal_combiner = SignalCombiner(
            strategy_weights={agent.name: 0.10 for agent in self.committee_agents},
            min_composite_score=0.65,
            min_consensus_ratio=0.50,
            min_voters=2,
        )

        # Universe Scanner
        self.scanner = UniverseScanner()

        self.scan_interval_seconds = scan_interval_seconds
        self.is_running = False

        # Persistent auto_trade, swing_mode, and shadow_mode state
        self.worker_state_file: Path = Path("data/quant_worker_state.json")
        worker_cfg = self._load_worker_config()
        self.auto_trade: bool = worker_cfg.get("auto_trade", True)
        self.close_positions_eod: bool = worker_cfg.get("close_positions_eod", False)
        self.shadow_mode: bool = worker_cfg.get("shadow_mode", False)

        # Pre-Trade Safety Guard & Holistic Risk Engine
        self.pre_trade_guard = PreTradeRiskGuard()

        # Dedicated Shadow Trading Engine (Zero broker orders, full forensic audit)
        self.shadow_engine = ShadowTradingEngine(
            initial_capital=float(worker_cfg.get("daily_starting_equity", 1000.0)),
            shadow_mode=self.shadow_mode,
        )

        # Harness² Engine (Google Research - Recursive Agent Harnessing)
        self.harness_engine = get_harness_engine()
        self._tracked_open_positions: Dict[str, Any] = {}

        # Always-On Throttle & Opening Plan Cache
        self._last_analysis_time: Optional[datetime] = None
        self.cached_candidates: list = []
        self.opening_plan: Dict[str, Any] = worker_cfg.get("opening_plan", {})

        # In-memory telemetry cache
        self.latest_telemetry: Dict[str, Any] = {}

        # Real-time activity logs for UI live engine feed
        self.activity_logs: list[dict[str, str]] = []
        self._add_log("Motor cuántico activo. Listo para analizar NASDAQ.", "info")

        # Dynamic Trailing Stop, Break-Even & Holding Tracker
        self.position_peaks: Dict[str, float] = dict(worker_cfg.get("position_peaks", {}))
        self.position_entry_times: Dict[str, str] = worker_cfg.get("position_entry_times", {})
        self.position_be_activated: Dict[str, bool] = worker_cfg.get("position_be_activated", {})
        self.position_partial_taken: Dict[str, bool] = worker_cfg.get("position_partial_taken", {})

        # Layer 2 Daily Equity Session Tracking
        self.session_date: str = worker_cfg.get("session_date", "")
        def_cap = float(getattr(self.settings.risk, "initial_capital", 100000.0))
        self.daily_starting_equity: float = float(worker_cfg.get("daily_starting_equity", def_cap))
        self.daily_peak_equity: float = float(worker_cfg.get("daily_peak_equity", self.daily_starting_equity))

        # Cooldown to avoid re-sending close orders every cycle while a previous one is still working
        self._exit_cooldown: Dict[str, datetime] = {}

        # Concentration & regime controls
        self.max_positions_per_sector: int = int(getattr(self.settings.risk, "max_positions_per_sector", 2))
        self._benchmark_cache: tuple[datetime, Optional[float], Optional[float]] | None = None

    async def _get_benchmark_trend(self, symbol: str = "QQQ") -> tuple[Optional[float], Optional[float]]:
        """Latest close and SMA200 of the benchmark (cached 1h). Returns (None, None) on any failure."""
        now = datetime.now(timezone.utc)
        if self._benchmark_cache and (now - self._benchmark_cache[0]).total_seconds() < 3600:
            return self._benchmark_cache[1], self._benchmark_cache[2]
        close = sma200 = None
        if self.settings.alpaca.validate_keys():
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                sec = self.settings.alpaca.secret_key
                raw_sec = sec.get_secret_value() if hasattr(sec, "get_secret_value") else str(sec)
                client = StockHistoricalDataClient(self.settings.alpaca.api_key, raw_sec)
                req = StockBarsRequest(
                    symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                    start=now - timedelta(days=420), limit=300,
                )
                bars = await asyncio.to_thread(client.get_stock_bars, req)
                closes = [float(b.close) for b in bars.data.get(symbol, [])]
                if len(closes) >= 200:
                    close = closes[-1]
                    sma200 = sum(closes[-200:]) / 200.0
            except Exception as exc:
                logger.debug("benchmark_trend_unavailable", error=str(exc))
        self._benchmark_cache = (now, close, sma200)
        return close, sma200

    async def _get_spread_pct(self, symbol: str, price: float) -> float:
        """Real bid/ask spread as a fraction of price (Alpaca latest quote); falls back to 5 bps."""
        fallback = 0.0005
        if price <= 0 or not self.settings.alpaca.validate_keys():
            return fallback
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            sec = self.settings.alpaca.secret_key
            raw_sec = sec.get_secret_value() if hasattr(sec, "get_secret_value") else str(sec)
            client = StockHistoricalDataClient(self.settings.alpaca.api_key, raw_sec)
            quotes = await asyncio.to_thread(
                client.get_stock_latest_quote, StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            )
            q = quotes.get(symbol)
            bid, ask = float(getattr(q, "bid_price", 0) or 0), float(getattr(q, "ask_price", 0) or 0)
            if bid > 0 and ask >= bid:
                return (ask - bid) / ((ask + bid) / 2.0)
        except Exception as exc:
            logger.debug("spread_quote_fallback", symbol=symbol, error=str(exc))
        return fallback

    def _load_worker_config(self) -> dict[str, Any]:
        def_cap = float(getattr(self.settings.risk, "initial_capital", 100000.0))
        if self.worker_state_file.exists():
            try:
                with open(self.worker_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    loaded_start = float(data.get("daily_starting_equity", def_cap))
                    loaded_peak = float(data.get("daily_peak_equity", def_cap))
                    return {
                        "auto_trade": bool(data.get("auto_trade", True)),
                        "close_positions_eod": bool(data.get("close_positions_eod", False)),
                        "shadow_mode": bool(data.get("shadow_mode", False)),
                        "position_entry_times": dict(data.get("position_entry_times", {})),
                        "position_be_activated": dict(data.get("position_be_activated", {})),
                        "position_partial_taken": dict(data.get("position_partial_taken", {})),
                        "position_peaks": dict(data.get("position_peaks", {})),
                        "session_date": str(data.get("session_date", "")),
                        "daily_starting_equity": loaded_start,
                        "daily_peak_equity": loaded_peak,
                        "opening_plan": dict(data.get("opening_plan", {})),
                    }
            except Exception:
                pass
        return {
            "auto_trade": True,
            "close_positions_eod": False,
            "shadow_mode": False,
            "position_entry_times": {},
            "position_be_activated": {},
            "position_partial_taken": {},
            "position_peaks": {},
            "session_date": "",
            "daily_starting_equity": def_cap,
            "daily_peak_equity": def_cap,
            "opening_plan": {},
        }

    def _save_worker_config(self) -> None:
        try:
            def_cap = float(getattr(self.settings.risk, "initial_capital", 100000.0))
            self.worker_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.worker_state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "auto_trade": self.auto_trade,
                    "close_positions_eod": self.close_positions_eod,
                    "shadow_mode": self.shadow_mode,
                    "position_entry_times": self.position_entry_times,
                    "position_be_activated": self.position_be_activated,
                    "position_partial_taken": self.position_partial_taken,
                    "position_peaks": getattr(self, "position_peaks", {}),
                    "session_date": getattr(self, "session_date", ""),
                    "daily_starting_equity": getattr(self, "daily_starting_equity", def_cap),
                    "daily_peak_equity": getattr(self, "daily_peak_equity", def_cap),
                    "opening_plan": getattr(self, "opening_plan", {}),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception:
            pass

    def set_shadow_mode(self, enabled: bool) -> None:
        """Sets shadow mode (simulate and record orders without submitting to broker)."""
        self.shadow_mode = enabled
        if hasattr(self, "shadow_engine"):
            self.shadow_engine.shadow_mode = enabled
        self._save_worker_config()
        mode_str = "ACTIVADO (Órdenes simuladas)" if enabled else "DESACTIVADO (Broker Real/Paper)"
        self._add_log(f"🧪 Shadow Mode {mode_str}", "info")

    def set_auto_trade(self, enabled: bool) -> None:

        """Sets auto_trade mode and persists the choice to disk."""
        self.auto_trade = enabled
        self._save_worker_config()

    def set_close_positions_eod(self, enabled: bool) -> None:
        """Sets end-of-day auto-flatten behavior (False = Swing Trading, True = Day Trading)."""
        self.close_positions_eod = enabled
        self._save_worker_config()
        mode_desc = "Day Trading (cierre a las 15:55 ET)" if enabled else "Swing Trading (posiciones abiertas overnight con protección GTC)"
        self._add_log(f"⚙️ Modo de sesión actualizado a: {mode_desc}", "info")

    def _add_log(self, message: str, level: str = "info") -> None:
        """Appends a timestamped activity event for the live web feed."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.activity_logs.append({"time": ts, "message": message, "level": level})
        if len(self.activity_logs) > 50:
            self.activity_logs.pop(0)

    def _find_bracket_levels(self, sym: str) -> tuple[Optional[float], Optional[float]]:
        """Extract active Stop-Loss and Take-Profit price levels for a symbol."""
        from services.broker_engine.order_state_machine import OrderStateMachine
        sl: Optional[float] = None
        tp: Optional[float] = None
        if hasattr(self.broker, "_orders") and hasattr(self.broker, "_order_requests"):
            for oid, ord_res in self.broker._orders.items():
                if ord_res.symbol == sym and OrderStateMachine.is_active(ord_res.status):
                    req = self.broker._order_requests.get(oid)
                    if req:
                        if req.stop_price is not None:
                            sl = req.stop_price
                        if req.limit_price is not None:
                            tp = req.limit_price
        if hasattr(self.broker, "_orders"):
            for ord_res in self.broker._orders.values():
                if ord_res.symbol != sym or not OrderStateMachine.is_active(ord_res.status):
                    continue
                raw = getattr(ord_res, "raw_response", {}) or {}
                if ord_res.order_type.value == "stop" and raw.get("stop_price") is not None:
                    sl = raw["stop_price"]
                elif ord_res.order_type.value == "limit" and raw.get("limit_price") is not None:
                    tp = raw["limit_price"]
        return sl, tp

    def _update_daily_session_risk(self, current_equity: float, cash: float, pos_count: int) -> None:
        """Tracks daily starting equity, daily peak equity, and feeds Layer 2 & Layer 3 into Circuit Breaker."""
        from datetime import datetime
        from core.market_session import ET

        now_et = datetime.now(ET)
        today_str = now_et.date().isoformat()

        if self.session_date != today_str:
            self.session_date = today_str
            self.daily_starting_equity = current_equity
            self.daily_peak_equity = current_equity
            self._save_worker_config()
            cb = get_circuit_breaker()
            cb.reset()
            self._add_log(f"🌅 Nueva sesión ({today_str}): Base diaria fijada en ${current_equity:,.2f}", "info")
        else:
            if current_equity > self.daily_peak_equity:
                self.daily_peak_equity = current_equity
                self._save_worker_config()

        cb = get_circuit_breaker()
        if hasattr(self.risk_engine, "get_or_create_portfolio_state"):
            state = self.risk_engine.get_or_create_portfolio_state(
                equity=current_equity,
                cash=cash,
                open_positions_count=pos_count,
                daily_starting_equity=self.daily_starting_equity,
                peak_equity=self.daily_peak_equity,
            )
            old_lvl = cb.level
            new_lvl = cb.evaluate(state)
            if new_lvl > old_lvl and new_lvl >= RiskLevel.STOP_NEW:
                trigger_msg = cb._trigger_reasons[-1] if cb._trigger_reasons else "Protección activada"
                self._add_log(f"🛡️ Candado de Riesgo ({new_lvl.name}): {trigger_msg}", "warning")

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Provides real-time telemetry state for API and dashboard."""
        cash = getattr(self.broker, "cash", 100000.0)
        positions = list(self.broker._positions.values()) if hasattr(self.broker, "_positions") else []
        portfolio_val = sum(p.market_value if p.side.lower() == "long" else -p.market_value for p in positions)
        total_equity = getattr(self.broker, "equity", cash + portfolio_val)
        buying_power = getattr(self.broker, "buying_power", max(0.0, cash * 2.0))
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        trade_history = getattr(self.broker, "get_trade_history", lambda: [])()
        realized_pnl = round(getattr(self.broker, "realized_pnl", 0.0), 2)
        realized_today = round(getattr(self.broker, "realized_pnl_today", 0.0), 2)
        initial_cap = getattr(self.broker, "initial_cash", self.settings.risk.initial_capital)
        # Acumulado exacto: equity = capital inicial + realizado + abierto (Alpaca paper sin comisiones).
        # No depende de la ventana de órdenes, así que no se infla por lotes sin pareja.
        if getattr(self.broker, "_connected", False):
            realized_pnl = round(total_equity - initial_cap - unrealized_pnl, 2)
        today_total_change = round(total_equity - getattr(self.broker, "last_equity", total_equity), 2)
        total_return_pct = round(
            ((total_equity - initial_cap) / initial_cap) * 100.0, 2
        ) if initial_cap > 0 else 0.0

        op_margin = float(getattr(self.settings.risk, "operational_capital", 1000.0))
        op_equity = round(op_margin + unrealized_pnl + realized_pnl, 2)
        long_val = sum(p.market_value for p in positions if p.side.lower() == "long")
        op_cash = round(max(0.0, op_margin - long_val), 2)
        op_peak_equity = max(op_margin, op_equity)
        op_return_pct = round(((op_equity - op_margin) / op_margin) * 100.0, 2)

        start_eq = getattr(self, "daily_starting_equity", initial_cap)
        if start_eq < total_equity * 0.1:
            start_eq = total_equity
        peak_eq = max(getattr(self, "daily_peak_equity", total_equity), total_equity)
        if peak_eq < total_equity * 0.1:
            peak_eq = total_equity
        daily_profit = max(0.0, total_equity - start_eq)
        peak_profit = max(0.0, peak_eq - start_eq)
        giveback_pct = round(((peak_profit - (total_equity - start_eq)) / peak_profit) * 100.0, 1) if peak_profit > 0 else 0.0

        ks = get_kill_switch()
        cb = get_circuit_breaker()
        profit_lock_active = bool(peak_profit >= cb.min_profit_lock_usd and cb.level >= RiskLevel.STOP_NEW)
        session_info = get_session_info()
        macro_features = self.regime_manager.get_macro_features()

        positions_data = []
        for p in positions:
            sl, tp = self._find_bracket_levels(p.symbol)
            entry_iso = self.position_entry_times.get(p.symbol)
            days_held = 0.0
            if entry_iso:
                try:
                    entry_dt = datetime.fromisoformat(entry_iso)
                    days_held = round((datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400.0, 1)
                except Exception:
                    pass

            positions_data.append({
                "symbol": p.symbol,
                "side": p.side.upper(),
                "qty": p.quantity,
                "entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "market_value": round(p.market_value, 2),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 2),
                "stop_loss": sl,
                "take_profit": tp,
                "break_even_active": self.position_be_activated.get(p.symbol, False),
                "days_open": days_held,
            })

        orders_data = []
        if hasattr(self.broker, "_orders"):
            for o in self.broker._orders.values():
                raw_resp = getattr(o, "raw_response", {}) or {}
                raw_status = raw_resp.get("raw_status") or o.status.value
                display_type = raw_resp.get("type_display") or o.order_type.value
                orders_data.append({
                    "order_id": o.broker_order_id,
                    "symbol": o.symbol,
                    "order_type": display_type,
                    "side": o.side.value,
                    "qty": o.quantity,
                    "status": raw_status,
                    "limit_price": raw_resp.get("limit_price"),
                    "stop_price": raw_resp.get("stop_price"),
                    "filled_price": o.filled_avg_price,
                    "created_at": o.submitted_at.isoformat() if o.submitted_at else "",
                })


        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker": {
                "running": self.is_running,
                "scan_interval_seconds": self.scan_interval_seconds,
                "broker": self.broker.name,
                "paper": self.broker.is_paper,
            },
            "trading_mode": self.settings.trading_mode.value,
            "auto_trade": self.auto_trade,
            "shadow_mode": self.shadow_mode,
            "shadow_metrics": self.shadow_engine.metrics.to_dict() if hasattr(self, "shadow_engine") else None,
            "close_positions_eod": self.close_positions_eod,
            "swing_mode": not self.close_positions_eod,
            "kill_switch_active": ks.is_active,
            "kill_switch_reason": ks.activation_reason,
            "circuit_breaker_tier": cb.level.name,
            "market_session": session_info,
            "opening_plan": getattr(self, "opening_plan", {}),

            "portfolio": {
                "cash": round(cash, 2),
                "portfolio_value": round(total_equity, 2),
                "buying_power": round(buying_power, 2),
                "realized_pnl": realized_pnl,
                "realized_pnl_today": realized_today,
                "today_total_change": today_total_change,
                "real_account_equity": round(total_equity, 2),
                "real_account_pnl_vs_initial": round(total_equity - initial_cap, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "total_return_pct": total_return_pct,
                "open_positions_count": len(positions_data),
                "daily_starting_equity": round(start_eq, 2),
                "daily_peak_equity": round(peak_eq, 2),
                "daily_peak_profit": round(peak_profit, 2),
                "profit_giveback_pct": giveback_pct,
                "profit_lock_active": profit_lock_active,
                "operational_margin": round(op_margin, 2),
                "operational_equity": round(op_equity, 2),
                "operational_cash": round(op_cash, 2),
                "operational_peak_equity": round(op_peak_equity, 2),
                "operational_return_pct": op_return_pct,
            },
            "positions": positions_data,
            "active_orders": orders_data,
            "trade_history": trade_history,
            "macro": macro_features,
            "latest_opportunities": self.latest_telemetry.get("top_candidates", []),
            "committee": {
                "agent_count": len(self.committee_agents),
                "agents": [agent.name for agent in self.committee_agents],
                "min_consensus_ratio": 0.60,
            },
            "news_scores": dict(self.latest_news_scores),
            "harness": self.harness_engine.get_summary() if hasattr(self, "harness_engine") else None,
            "activity_logs": list(reversed(self.activity_logs[-15:])),
            "latest_status": self.activity_logs[-1]["message"] if self.activity_logs else "Sistema en espera",
        }

    async def validate_pending_premarket_orders(self) -> None:
        """Guardián de Pre-Apertura (Pre-Market Order Sentinel).
        
        Evaluates working / queued orders waiting for market open:
        1. Checks if SEC EDGAR issued an adverse filing block for the ticker.
        2. Checks if macro regime risk multiplier dropped to 0.0x.
        3. Analyzes pre-market price action and adverse overnight gaps:
           - For BUY orders: If pre-market price dropped below planned Stop-Loss, cancel immediately.
           - For SELL orders: If pre-market price surged above planned Stop-Loss, cancel immediately.
        """
        if not hasattr(self.broker, "_orders") or not self.broker._orders:
            return

        orders_to_check = [
            o for o in self.broker._orders.values()
            if str((getattr(o, "raw_response", {}) or {}).get("raw_status") or o.status.value).lower() in ["accepted", "new", "submitted"]
            and o.order_type.value == "market"
        ]

        if not orders_to_check:
            return

        alpaca_prices: dict[str, float] = {}
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestBarRequest
            s = self.settings
            raw_sec = s.alpaca.secret_key.get_secret_value() if hasattr(s.alpaca.secret_key, "get_secret_value") else str(s.alpaca.secret_key)
            data_client = StockHistoricalDataClient(s.alpaca.api_key, raw_sec)
            symbols = list({o.symbol for o in orders_to_check})
            bars = data_client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=symbols))
            for sym, b in bars.items():
                if b and b.close:
                    alpaca_prices[sym] = float(b.close)
        except Exception as exc:
            logger.debug("premarket_data_fetch_fallback", error=str(exc))

        cancelled_count = 0
        for o in orders_to_check:
            sym = o.symbol
            cancel_reason: Optional[str] = None

            # 1. SEC EDGAR adverse filings
            if self.regime_manager.is_symbol_blocked(sym):
                cancel_reason = "Reporte adverso en SEC EDGAR detectado"

            # 2. Macro Risk Halt
            elif self.regime_manager.get_risk_multiplier() <= 0.0:
                cancel_reason = "Régimen Macro degradado a Riesgo 0.0x"

            # 3. Overnight Gap / Price Action Check against Stop-Loss
            else:
                current_p = alpaca_prices.get(sym)
                if not current_p:
                    fv = self.scanner.feature_store.get(sym)
                    current_p = getattr(fv, "close_price", None)

                if current_p and current_p > 0:
                    stop_p = None
                    req = getattr(self.broker, "_order_requests", {}).get(o.broker_order_id)
                    if req and req.stop_loss_price:
                        stop_p = req.stop_loss_price
                    else:
                        for leg in self.broker._orders.values():
                            if leg.symbol == sym and leg.order_type.value == "stop":
                                stop_p = (getattr(leg, "raw_response", {}) or {}).get("stop_price")
                                if stop_p:
                                    break

                    is_buy = o.side.value.lower() == "buy"
                    if is_buy and stop_p and current_p <= stop_p:
                        cancel_reason = f"Gap bajista adverso en Pre-Market (${current_p:.2f} <= Stop ${stop_p:.2f})"
                    elif not is_buy and stop_p and current_p >= stop_p:
                        cancel_reason = f"Gap alcista adverso en Pre-Market (${current_p:.2f} >= Stop ${stop_p:.2f})"

            if cancel_reason:
                try:
                    await self.broker.cancel_order(o.broker_order_id)
                    cancelled_count += 1
                    logger.warning("premarket_order_cancelled_by_sentinel", symbol=sym, order_id=o.broker_order_id, reason=cancel_reason)
                    self._add_log(f"🛡️ Guardián Pre-Apertura: Cancelada orden en cola de {sym} ({cancel_reason})", "warning")
                except Exception as exc:
                    logger.error("premarket_order_cancel_error", symbol=sym, error=str(exc))

        if cancelled_count > 0:
            await self.broker.reconcile()

    async def refresh_news_scores(self, symbols: list[str]) -> dict[str, float]:
        """Fetch and score recent news via Alpaca + Gemini AI; return symbol scores for the trade gate."""
        if not symbols:
            self.latest_news_scores = {}
            return {}
        try:
            if self.news_adapter is not None:
                articles = await self.news_adapter.get_latest_news(
                    symbols=symbols[:50],
                    limit=50,
                    start=datetime.now(timezone.utc) - timedelta(hours=24),
                )
                for article in articles:
                    if hasattr(self.news_engine, "process_article_async"):
                        await self.news_engine.process_article_async(article)
                    else:
                        self.news_engine.process_article(article)
            scores = {
                symbol: self.news_engine.get_symbol_sentiment(symbol).composite_news_score
                for symbol in symbols
            }
            self.latest_news_scores = scores
            return scores
        except Exception as exc:
            logger.warning("news_refresh_failed", error=str(exc))
            self.latest_news_scores = {}
            return {}

    async def execute_cycle(self, auto_trade: Optional[bool] = None) -> Dict[str, Any]:
        """Executes a single end-to-end quant scan and trading evaluation cycle.
        
        Args:
            auto_trade: If True, executes qualifying signals into broker orders.
                       If False, performs pure market scanning, ML veto filtering, and radar ranking without order entry.
                       If None, defaults to self.auto_trade.
        """
        should_trade = self.auto_trade if auto_trade is None else auto_trade
        now_et = datetime.now(ET)
        caps = get_session_capabilities(now_et)
        session = caps.session
        ks = get_kill_switch()

        logger.info(
            "quant_cycle_start",
            session=session.value,
            granular_session=caps.granular_session,
            kill_switch=ks.is_active,
            auto_trade=should_trade,
            positions=len(self.broker._positions),
        )

        # 0. Ensure broker is connected and synchronized
        if hasattr(self.broker, "connect") and not getattr(self.broker, "_connected", False):
            await self.broker.connect()
        elif hasattr(self.broker, "reconcile"):
            await self.broker.reconcile()

        # 0.4. Capa 2 & Capa 3: Actualizar tracking diario de equity y evaluar Circuit Breaker
        cash_val = getattr(self.broker, "cash", 100000.0)
        pos_list = list(self.broker._positions.values())
        pf_val = sum(p.market_value if p.side.lower() == "long" else -p.market_value for p in pos_list)
        current_eq = getattr(self.broker, "equity", cash_val + pf_val)
        self._update_daily_session_risk(current_eq, cash_val, len(pos_list))

        # 0.5. Guardián de Pre-Apertura: Validar órdenes en cola contra noticias y gaps adversos
        await self.validate_pending_premarket_orders()

        # 1. Kill Switch Check
        if ks.is_active:
            logger.warning("cycle_skipped_kill_switch_active", reason=ks.activation_reason)
            self._add_log(f"🛑 Escaneo detenido: Kill Switch activo ({ks.activation_reason})", "error")
            return self.get_telemetry_snapshot()

        # 2. Decoupled Market Session Capabilities Gate
        if not caps.can_enter:
            should_trade = False

        is_extended_session = caps.is_premarket or caps.is_afterhours
        is_regular_session = caps.is_regular
        is_weekend_or_overnight = caps.granular_session in {"weekend_analysis", "overnight_analysis", "holiday_analysis"}

        if is_weekend_or_overnight:
            now_utc = datetime.now(timezone.utc)
            seconds_since_last_scan = (
                (now_utc - self._last_analysis_time).total_seconds()
                if self._last_analysis_time else 999999.0
            )
            need_scan = seconds_since_last_scan >= caps.recommended_cycle_seconds or not self.cached_candidates

            if not need_scan:
                mins_left = int(max(1, (caps.recommended_cycle_seconds - seconds_since_last_scan) / 60))
                self._add_log(
                    f"🧠 Modo Always-On ({caps.granular_session}): Radar de apertura activo · próxima re-evaluación en {mins_left}m",
                    "info",
                )
                return self.get_telemetry_snapshot()
            else:
                self._add_log(f"🧠 Modo Always-On ({caps.granular_session}): Ejecutando escaneo y plan de apertura...", "info")
                self._last_analysis_time = now_utc

        if is_extended_session:
            logger.info("extended_hours_trading_active", session=session.value, granular=caps.granular_session)
            self._add_log(f"🌙 Sesión Extendida Activa ({caps.granular_session}) · Modo Órdenes Limitadas con Catalizador", "info")

        # 3. Macro Regime Scaling & Adverse Checks
        macro_risk_mult = self.regime_manager.get_risk_multiplier()
        # 3.5. Gestión de Fin de Jornada (Risk Governor Progresivo):
        is_eod_window = (now_et.weekday() < 5) and (now_et.hour == 15 and now_et.minute >= 55)
        # Bloquear apertura de nuevas posiciones si la sesión está en Near Close o Risk Descalation
        if caps.is_near_close:
            should_trade = False


        if is_eod_window:
            if self.close_positions_eod:
                # Modo Intradía Estricto: liquidar todo a las 15:55 ET si el usuario lo configuró explícitamente
                if hasattr(self.broker, "_positions") and self.broker._positions:
                    self._add_log("🔔 FIN DEL DÍA (15:55 ET): Modo Intradía activo — cerrando posiciones para asegurar caja.", "info")
                    logger.info("end_of_day_auto_flatten_triggered", positions=len(self.broker._positions))
                    try:
                        if hasattr(self.broker, "client"):
                            self.broker.client.cancel_orders()
                            pos_list = self.broker.client.get_all_positions()
                            from alpaca.trading.requests import MarketOrderRequest
                            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce as AlpacaTIF
                            for p in pos_list:
                                qty = abs(int(float(p.qty)))
                                side = AlpacaSide.BUY if float(p.qty) < 0 else AlpacaSide.SELL
                                req = MarketOrderRequest(symbol=p.symbol, qty=str(qty), side=side, time_in_force=AlpacaTIF.DAY)
                                self.broker.client.submit_order(req)
                        self.position_peaks.clear()
                        await self.broker.reconcile()
                        self._add_log("✅ Todas las posiciones cerradas por fin de día (Modo Intradía).", "success")
                    except Exception as eod_err:
                        logger.error("eod_flatten_error", error=str(eod_err))
            else:
                # Modo Swing Trading (Recomendado): Mantener posiciones abiertas overnight con órdenes GTC
                if hasattr(self.broker, "_positions") and self.broker._positions:
                    if not getattr(self, "_eod_logged_today", False):
                        self._add_log(
                            f"🌙 FIN DE SESIÓN REGULAR (15:55 ET): Modo Swing Trading activo — {len(self.broker._positions)} posición(es) se mantienen abiertas overnight con protección GTC.",
                            "info",
                        )
                        self._eod_logged_today = True
        else:
            self._eod_logged_today = False

        # 4. Scan Universe for High-Probability Setups
        self._add_log("⚡ Conectando a Twelve Data... Descargando velas de NASDAQ", "info")
        universe = UniverseManager.get_universe("nasdaq")
        top_candidates = await self.scanner.scan_universe(symbols=universe, top_n=5)

        top_names = ", ".join([f"{c.symbol} ({c.composite_score:.0f})" for c in top_candidates])
        self._add_log(f"📊 Análisis completado (Supertrend, FVG, EMA). Top: {top_names}", "success")

        self.latest_telemetry["top_candidates"] = [
            {
                "symbol": c.symbol,
                "score": round(c.composite_score, 1),
                "setup": c.setup_type,
                "direction": c.direction,
            }
            for c in top_candidates
        ]
        tracked_news_syms = list(dict.fromkeys([c.symbol for c in top_candidates] + list(getattr(self.broker, '_positions', {}).keys())))
        news_scores = await self.refresh_news_scores(tracked_news_syms)

        # If weekend, overnight, or holiday: generate and persist Opening Plan, then finish analysis cycle
        if is_weekend_or_overnight:
            self.cached_candidates = top_candidates
            self.opening_plan = {
                "session": caps.granular_session,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "top_candidates": [
                    {
                        "symbol": c.symbol,
                        "score": round(c.composite_score, 1),
                        "direction": c.direction if isinstance(c.direction, str) else getattr(c.direction, "name", str(c.direction)),
                        "setup": c.setup_type,
                        "news_score": news_scores.get(c.symbol, 0.0),
                    }
                    for c in top_candidates
                ],
                "valid_until": "09:35 ET",
            }
            self.latest_telemetry["opening_plan"] = self.opening_plan
            self._save_worker_config()
            self._add_log(f"📋 Plan de Apertura generado: {len(top_candidates)} setups en radar para la próxima campana.", "success")
            return self.get_telemetry_snapshot()

        # 4.5. Mark-to-Market & Bracket Execution for Open Positions

        # Feeds latest real market price action into existing positions so Take-Profit or Stop-Loss orders can fill accurately
        for p_sym, pos in list(self.broker._positions.items()):
            fv = self.scanner.feature_store.get(p_sym)
            base_p = getattr(fv, "close_price", None) or pos.current_price
            p_high = getattr(fv, "high_price", None) or base_p
            p_low = getattr(fv, "low_price", None) or base_p
            p_close = base_p

            if hasattr(self.broker, "update_market_price"):
                self.broker.update_market_price(p_sym, p_close, high=p_high, low=p_low)

        # 5. Evaluate Opportunities through Strategies & Risk Gates
        cb = get_circuit_breaker()
        if should_trade and not cb.can_open_new_positions:
            should_trade = False
            self._add_log(f"🛡️ Nuevas compras pausadas por Circuit Breaker ({cb.level.name})", "warning")

        if should_trade:
            self._add_log(f"⚡ Análisis Multi-Timeframe / Diario: {len(top_candidates)} setups evaluados · Auto-Trade AUTOMÁTICO", "info")
            for candidate in top_candidates:
                sym = candidate.symbol
                news_score = news_scores.get(sym, 0.0)
                if news_score <= -50.0:
                    logger.warning("symbol_blocked_by_adverse_news", symbol=sym, news_score=news_score)
                    self._add_log(
                        f"📰 {sym} bloqueado por noticias adversas (score {news_score:.1f})",
                        "warning",
                    )
                    continue

                # En Horario Extendido (Pre-Market / After-Hours), solo operar si existe un catalizador de noticias significativo
                if is_extended_session and abs(news_score) < 30.0:
                    logger.info("extended_hours_skipped_no_catalyst", symbol=sym, news_score=news_score)
                    continue

                # Check if SEC EDGAR has an adverse filing block for this ticker
                if self.regime_manager.is_symbol_blocked(sym):
                    logger.warning("symbol_blocked_by_sec_edgar", symbol=sym)
                    continue

                # Don't double-allocate if already in position OR has pending/held orders in broker
                active_symbols = set(self.broker._positions.keys())
                if hasattr(self.broker, "_orders"):
                    for o in self.broker._orders.values():
                        raw_st = str((getattr(o, "raw_response", {}) or {}).get("raw_status") or o.status.value).lower()
                        if raw_st in ["accepted", "held", "new", "pending_new", "submitted", "partially_filled"]:
                            active_symbols.add(o.symbol)

                if sym in active_symbols:
                    logger.info("symbol_already_active_or_pending", symbol=sym)
                    continue

                # Only execute clear directional setups (bullish / bearish), skip neutral
                if candidate.direction not in ["bullish", "bearish"]:
                    logger.info("symbol_skipped_neutral_bias", symbol=sym, direction=candidate.direction)
                    continue

                # Check if feature vector is available
                if not candidate.features:
                    continue

                f_vec = candidate.features.to_ml_features()
                should_veto = self.ml_predictor.should_veto_signal(
                    symbol=sym,
                    signal_direction="BUY" if candidate.direction == "bullish" else "SELL",
                    feature_vector=f_vec,
                )
                if should_veto:
                    logger.info("trade_vetoed_by_ml_model", symbol=sym, direction=candidate.direction)
                    if hasattr(self, "shadow_engine"):
                        self.shadow_engine.metrics.rejections_by_guard["MLPredictor"] = (
                            self.shadow_engine.metrics.rejections_by_guard.get("MLPredictor", 0) + 1
                        )
                        self.shadow_engine.metrics.rejection_reasons.append({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "symbol": sym,
                            "guard": "MLPredictor",
                            "reason": "Vetoed by ML model probability threshold",
                        })
                    continue

                current_price = candidate.features.close_price
                spread_pct_real = await self._get_spread_pct(sym, current_price)
                spread = current_price * spread_pct_real
                if hasattr(self.broker, "update_market_price"):
                    self.broker.update_market_price(sym, current_price)

                from strategies.base import SignalDirection
                from services.strategy_engine.signal_combiner import CombinedSignal
                direction = SignalDirection.BUY if candidate.direction == "bullish" else SignalDirection.SELL
                norm_score = min(1.0, max(0.0, candidate.composite_score / 100.0))

                # ── 6. Run the ten-agent committee (enriched with News & Gemini features) ──
                features_dict = candidate.features.to_dict()
                if hasattr(self.news_engine, "to_feature_dict"):
                    features_dict.update(self.news_engine.to_feature_dict(sym))
                else:
                    features_dict["news_composite_score"] = news_score
                market_regime = getattr(self.regime_manager, "current_regime", None)
                try:
                    strategy_signals = await asyncio.gather(*(
                        agent.analyze(sym, features_dict, market_regime)
                        for agent in self.committee_agents
                    ))
                    valid_signals = [s for s in strategy_signals if s is not None]
                    if hasattr(self, "harness_engine"):
                        self.signal_combiner._weights = self.harness_engine.get_adjusted_committee_weights(
                            self.signal_combiner.DEFAULT_WEIGHTS
                        )
                    ensemble = self.signal_combiner.combine(
                        sym, valid_signals, market_regime, news_composite_score=news_score
                    )

                    if valid_signals and ensemble.is_approved and ensemble.direction == direction:
                        # Strategy ensemble confirmed the scanner's directional bias → use ensemble
                        combined_signal = CombinedSignal(
                            symbol=sym,
                            direction=direction,
                            composite_score=max(norm_score, ensemble.composite_score),
                            confidence=ensemble.confidence,
                            consensus_ratio=ensemble.consensus_ratio,
                            contributing_strategies=ensemble.contributing_strategies + ["CompositeOpportunityEngine"],
                            is_approved=True,
                            reasons=candidate.reasons[:2] + ensemble.reasons + [
                                f"News composite score: {news_score:.1f}"
                            ],
                            suggested_stop=ensemble.suggested_stop or round(
                                current_price * (0.98 if direction == SignalDirection.BUY else 1.02), 2
                            ),
                            suggested_target=ensemble.suggested_target or round(
                                current_price * (1.04 if direction == SignalDirection.BUY else 0.96), 2
                            ),
                        )
                        logger.info(
                            "strategy_ensemble_confirmed",
                            symbol=sym,
                            strategies=ensemble.contributing_strategies,
                            score=ensemble.composite_score,
                        )
                    else:
                        logger.info(
                            "strategy_committee_no_consensus_no_trade",
                            symbol=sym,
                            valid_signals=len(valid_signals),
                            ensemble_direction=ensemble.direction.value,
                            scanner_direction=direction.value,
                            consensus_ratio=ensemble.consensus_ratio,
                        )
                        if hasattr(self, "shadow_engine"):
                            self.shadow_engine.metrics.rejections_by_guard["CommitteeNoConsensus"] = (
                                self.shadow_engine.metrics.rejections_by_guard.get("CommitteeNoConsensus", 0) + 1
                            )
                            self.shadow_engine.metrics.rejection_reasons.append({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "symbol": sym,
                                "guard": "CommitteeNoConsensus",
                                "reason": f"Insufficient consensus ratio: {ensemble.consensus_ratio:.2f} < 0.60 or directional mismatch",
                            })
                        continue
                except Exception as strat_err:
                    logger.warning("strategy_ensemble_eval_error", symbol=sym, error=str(strat_err))
                    continue

                # 6.5. Pre-Trade Risk Guard (Multidimensional Gatekeeper)
                fv = self.scanner.feature_store.get(sym)
                natr = getattr(fv, "natr_14", None) if fv else None
                # Use true overnight/intraday reference close when available; during regular session
                # dampen full daily bar return so strong trend/breakout candles are not misclassified as adverse gaps
                ret_1 = getattr(candidate.features, "return_1", 0.0) or 0.0
                effective_gap_ret = ret_1 if is_extended_session else (ret_1 * 0.35)
                prev_c = current_price / (1.0 + effective_gap_ret / 100.0) if (1.0 + effective_gap_ret / 100.0) > 0 else current_price

                guard_res = self.pre_trade_guard.evaluate_order_proposal(
                    symbol=sym,
                    direction="buy" if direction == SignalDirection.BUY else "sell",
                    current_price=current_price,
                    prev_close=prev_c,
                    spread_pct=spread / current_price if current_price > 0 else 0.001,
                    natr_14=natr,
                    news_score=news_score,
                    macro_risk_mult=macro_risk_mult,
                    is_exit=False,
                    capabilities=caps,
                )
                if not guard_res.approved:
                    logger.warning("signal_rejected_by_pre_trade_guard", symbol=sym, violations=guard_res.violations)
                    self._add_log(f"🛡️ Guardián Pre-Trade ({sym}): {guard_res.violations[0]}", "warning")
                    if self.shadow_mode and hasattr(self, "shadow_engine"):
                        self.shadow_engine.evaluate_and_record_decision(
                            symbol=sym,
                            strategies=combined_signal.contributing_strategies,
                            direction=direction.name,
                            action="ENTRY",
                            session_name=session.name,
                            signal_score=combined_signal.composite_score,
                            base_quantity=25.0,
                            session_multiplier=caps.max_position_size_multiplier if caps else 1.0,
                            gap_multiplier=guard_res.size_multiplier,
                            final_quantity=0.0,
                            spread_pct=spread / current_price if current_price > 0 else 0.001,
                            natr_14=natr,
                            price=current_price,
                            order_type="LIMIT" if is_extended_session else "MARKET",
                            stop_price=combined_signal.suggested_stop,
                            target_price=combined_signal.suggested_target,
                            risk_approved=False,
                            rejection_reason="; ".join(guard_res.violations),
                            signal_timestamp=getattr(combined_signal, "timestamp", None),
                            guard_name="PreTradeRiskGuard",
                        )
                    continue

                for warn_msg in guard_res.warnings:
                    self._add_log(f"⚠️ Alerta Pre-Trade ({sym}): {warn_msg}", "warning")

                # 6.8. Sector exposure cap (hidden correlation guard)
                from services.risk_engine.position_sizing import check_sector_exposure, market_trend_multiplier
                sector_chk = check_sector_exposure(sym, active_symbols, max_per_sector=self.max_positions_per_sector)
                if not sector_chk.allowed:
                    self._add_log(f"🧩 {sym} omitido: {sector_chk.reason}", "warning")
                    logger.info("entry_blocked_by_sector_cap", symbol=sym, sector=sector_chk.sector, open=sector_chk.open_in_sector)
                    continue

                # 6.9. Benchmark trend sizing (QQQ vs SMA200): reduce size when trading against the regime
                bench_close, bench_sma200 = await self._get_benchmark_trend()
                trend_mult = market_trend_multiplier(bench_close, bench_sma200, is_long=(direction == SignalDirection.BUY))
                entry_size_mult = guard_res.size_multiplier * trend_mult
                if trend_mult < 1.0:
                    self._add_log(f"📉 {sym}: operación contra la tendencia de QQQ (SMA200) · tamaño ×{trend_mult:.1f}", "info")

                # 7. Route through Execution Engine or Shadow Mode
                if self.shadow_mode:
                    import math
                    settings = getattr(self, "settings", None) or get_settings()
                    use_op_cap = getattr(settings.risk, "use_operational_capital", True)
                    real_cap = float(getattr(settings.risk, "operational_capital", 1000.0)) if use_op_cap else float(getattr(self.broker, "equity", settings.risk.initial_capital))
                    max_sz = float(getattr(settings.risk, "max_operational_position_size", 0.20)) if use_op_cap else float(getattr(settings.risk, "max_position_size", 0.05))
                    t_notional = real_cap * max_sz
                    base_shs = max(1.0, float(math.floor(t_notional / current_price))) if (current_price > 0 and current_price <= real_cap * 0.45) else (1.0 if (current_price > 0 and current_price <= real_cap * 0.45) else 0.0)
                    final_shs = float(math.floor(base_shs * entry_size_mult))
                    exec_limit_price = round(current_price * (1.002 if direction == SignalDirection.BUY else 0.998), 2) if is_extended_session else None

                    shadow_rec = self.shadow_engine.evaluate_and_record_decision(
                        symbol=sym,
                        strategies=combined_signal.contributing_strategies,
                        direction=direction.name,
                        action="ENTRY",
                        session_name=session.name,
                        signal_score=combined_signal.composite_score,
                        base_quantity=base_shs,
                        session_multiplier=caps.max_position_size_multiplier if caps else 1.0,
                        gap_multiplier=guard_res.size_multiplier,
                        final_quantity=final_shs,
                        spread_pct=spread / current_price if current_price > 0 else 0.001,
                        natr_14=natr,
                        price=current_price,
                        order_type="LIMIT" if is_extended_session else "MARKET",
                        limit_price=exec_limit_price,
                        stop_price=combined_signal.suggested_stop,
                        target_price=combined_signal.suggested_target,
                        risk_approved=True,
                        signal_timestamp=getattr(combined_signal, "timestamp", None),
                    )
                    self._add_log(f"🧪 [SHADOW MODE] Orden simulada para {sym} ({direction.name}) @ ${current_price:.2f} · ID: {shadow_rec.client_order_id}", "info")
                    logger.info("shadow_mode_order_simulated", symbol=sym, direction=direction.name, price=current_price, client_order_id=shadow_rec.client_order_id)
                    order_response = None
                elif is_extended_session:
                    # Política de ejecución: órdenes límite para horario extendido
                    exec_limit_price = round(current_price * (1.002 if direction == SignalDirection.BUY else 0.998), 2)
                    order_response = await self.execution_engine.execute_signal(
                        signal=combined_signal,
                        current_price=current_price,
                        current_spread_pct=spread / current_price,
                        extended_hours=True,
                        limit_price=exec_limit_price,
                        natr_14=natr,
                        size_multiplier=entry_size_mult,
                        sector=sector_chk.sector,
                    )
                else:
                    order_response = await self.execution_engine.execute_signal(
                        signal=combined_signal,
                        current_price=current_price,
                        current_spread_pct=spread / current_price,
                        natr_14=natr,
                        size_multiplier=entry_size_mult,
                        sector=sector_chk.sector,
                    )


                if order_response:
                    order_status = getattr(order_response, "status", None)
                    is_filled = str(getattr(order_status, "value", order_status)).lower() in {"filled", "partially_filled"}
                    logger.info(
                        "quant_order_filled" if is_filled else "quant_order_submitted",
                        symbol=sym,
                        direction=direction.name,
                        status=str(getattr(order_status, "value", order_status)),
                        order_id=getattr(order_response, "broker_order_id", getattr(order_response, "order_id", "")),
                    )
                    if is_filled:
                        self._add_log(f"🚀 Orden ejecutada para {sym} ({direction.name})", "success")
                    else:
                        self._add_log(f"📨 Orden enviada para {sym} ({direction.name}) · pendiente de fill", "info")
        else:
            logger.info("quant_scan_completed_analysis_only", candidates_found=len(top_candidates))
            self._add_log(f"🛡️ Análisis Multi-Timeframe / Diario completado: {len(top_candidates)} setups en radar · Auto-Trade OFF (Solo Análisis)", "info")

        # 7.5. Software Stop-Loss Monitor (Alpaca paper constraint: only 1 closing order per symbol)
        # Checks every cycle if any position has breached its SL threshold.
        # SL levels: LONG = entry * 0.97 (-3%), SHORT = entry * 1.03 (+3%)
        try:
            from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
            from alpaca.trading.enums import QueryOrderStatus, OrderSide as AlpacaSide, TimeInForce as AlpacaTIF
            if hasattr(self.broker, "_positions") and hasattr(self.broker, "client"):
                open_orders_resp = self.broker.client.get_orders(
                    GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50)
                )
                # Map symbol -> open order IDs (to cancel before SL close)
                sym_to_order_ids: Dict[str, list] = {}
                for o in open_orders_resp:
                    sym_to_order_ids.setdefault(o.symbol, []).append(str(o.id))

                # Limpieza de tracking para posiciones que ya no existen en el broker
                current_active_syms = set(self.broker._positions.keys())
                for sym_t in list(self.position_entry_times.keys()):
                    if sym_t not in current_active_syms:
                        self.position_entry_times.pop(sym_t, None)
                        self.position_be_activated.pop(sym_t, None)
                        self.position_partial_taken.pop(sym_t, None)
                        self.position_peaks.pop(sym_t, None)

                for p_sym, pos in list(self.broker._positions.items()):
                    mark = float(pos.current_price)
                    entry = float(pos.avg_entry_price)
                    is_long = str(pos.side).lower() == "long"

                    # 0. Registrar fecha de apertura si es nueva posición detectada
                    if p_sym not in self.position_entry_times:
                        self.position_entry_times[p_sym] = datetime.now(timezone.utc).isoformat()
                        self._save_worker_config()

                    # 1. Registrar Pico Máximo para Trailing Stop ("Trancar en Ganancia")
                    if is_long:
                        peak = max(self.position_peaks.get(p_sym, mark), mark)
                        self.position_peaks[p_sym] = peak
                        gain_pct = (mark - entry) / entry if entry > 0 else 0.0
                        peak_gain_pct = (peak - entry) / entry if entry > 0 else 0.0
                    else:
                        trough = min(self.position_peaks.get(p_sym, mark), mark)
                        self.position_peaks[p_sym] = trough
                        gain_pct = (entry - mark) / entry if entry > 0 else 0.0
                        peak_gain_pct = (entry - trough) / entry if entry > 0 else 0.0

                    # 2. Niveles de Seguridad Base (Bracket o ATR dinámico)
                    sl_level, tp_level = self._find_bracket_levels(p_sym)
                    if sl_level is None or tp_level is None:
                        fv = self.scanner.feature_store.get(p_sym)
                        natr = getattr(fv, "natr_14", None) if fv else None
                        atr_buffer = (float(natr) / 100.0 * 1.5) if natr else 0.030
                        atr_buffer = max(0.025, min(0.045, atr_buffer))  # 2.5% a 4.5% según volatilidad real
                        if sl_level is None:
                            sl_level = round(entry * (1.0 - atr_buffer if is_long else 1.0 + atr_buffer), 2)
                        if tp_level is None:
                            # Objetivo de Take-Profit realista de Swing: 1.5R (aprox +3.5% a +5.0%)
                            tp_level = round(entry * (1.0 + atr_buffer * 1.5 if is_long else 1.0 - atr_buffer * 1.5), 2)

                    initial_sl = sl_level  # stop before any break-even adjustment (defines 1R)

                    # 3. BREAK-EVEN DINÁMICO (Protección Cero Riesgo):
                    # Si el precio avanza >= 45% hacia el Take-Profit o supera +1.0% de ganancia,
                    # se eleva el Stop Loss al precio de entrada (+0.3% para comisiones).
                    halfway_to_tp = False
                    if tp_level:
                        tp_dist = abs(tp_level - entry)
                        if is_long and mark >= (entry + tp_dist * 0.45):
                            halfway_to_tp = True
                        elif not is_long and mark <= (entry - tp_dist * 0.45):
                            halfway_to_tp = True

                    # Ceñido por Break-Even estándar (+1% o 45% a TP) o por Capa 2 (Profit Lock activo)
                    force_profit_lock_be = bool(cb.level >= RiskLevel.STOP_NEW and gain_pct >= 0.005)

                    if peak_gain_pct >= 0.010 or halfway_to_tp or force_profit_lock_be:
                        be_price = round(entry * (1.003 if is_long else 0.997), 2)
                        if is_long and be_price > sl_level:
                            sl_level = be_price
                            if not self.position_be_activated.get(p_sym, False):
                                self.position_be_activated[p_sym] = True
                                log_prefix = "🛡️ CAPA 2 (PROFIT LOCK)" if force_profit_lock_be else "🛡️ BREAK-EVEN"
                                self._add_log(f"{log_prefix} ACTIVADO: {p_sym} avanzó +{gain_pct*100:.2f}%. Stop elevado a entrada ${sl_level:.2f} (Cero Riesgo).", "success")
                        elif not is_long and be_price < sl_level:
                            sl_level = be_price
                            if not self.position_be_activated.get(p_sym, False):
                                self.position_be_activated[p_sym] = True
                                log_prefix = "🛡️ CAPA 2 (PROFIT LOCK)" if force_profit_lock_be else "🛡️ BREAK-EVEN"
                                self._add_log(f"{log_prefix} ACTIVADO: {p_sym} avanzó +{gain_pct*100:.2f}%. Stop bajado a entrada ${sl_level:.2f} (Cero Riesgo).", "success")

                    sl_breached = (mark <= sl_level) if is_long else (mark >= sl_level)
                    tp_reached = (mark >= tp_level) if is_long else (mark <= tp_level)

                    close_reason = None
                    if tp_reached or peak_gain_pct >= 0.040:
                        close_reason = f"🎯 TAKE-PROFIT ALCANZADO: {p_sym} {'LONG' if is_long else 'SHORT'} mark=${mark:.2f} objetivo=${tp_level:.2f} (+{abs(gain_pct)*100:.2f}%)"
                    elif sl_breached:
                        close_reason = f"🛑 STOP-LOSS TÉCNICO: {p_sym} {'LONG' if is_long else 'SHORT'} mark=${mark:.2f} stop=${sl_level:.2f} ({gain_pct*100:.2f}%)"
                    elif peak_gain_pct >= 0.015:
                        # Trailing Stop: asegurar mínimo +0.4% neto y permitir solo 1.0% de retroceso desde el pico
                        if is_long:
                            trail_sl = max(round(entry * 1.004, 2), round(peak * 0.990, 2))
                            if mark <= trail_sl:
                                close_reason = f"🔒 TRANCA EN GANANCIA: {p_sym} LONG protegió beneficio a ${mark:.2f} tras pico en ${peak:.2f} (+{gain_pct*100:.2f}%)"
                        else:
                            trail_sl = min(round(entry * 0.996, 2), round(trough * 1.010, 2))
                            if mark >= trail_sl:
                                close_reason = f"🔒 TRANCA EN GANANCIA: {p_sym} SHORT protegió beneficio a ${mark:.2f} tras mínimo en ${trough:.2f} (+{gain_pct*100:.2f}%)"

                    # 3.5. STOP REAL EN EL BROKER: si la posición no tiene ninguna orden de salida
                    # abierta en Alpaca, colocar un stop GTC para que la protección no dependa del motor.
                    if not close_reason and not sym_to_order_ids.get(p_sym):
                        try:
                            from alpaca.trading.requests import StopOrderRequest
                            stop_ts = getattr(self, "_broker_stop_ts", None)
                            if stop_ts is None:
                                stop_ts = self._broker_stop_ts = {}
                            last_stop = stop_ts.get(p_sym)
                            safe_side = (sl_level < mark) if is_long else (sl_level > mark)
                            stop_qty = abs(int(pos.quantity))
                            if (
                                safe_side and stop_qty > 0 and not is_extended_session
                                and not (last_stop and (datetime.now(timezone.utc) - last_stop).total_seconds() < 300)
                            ):
                                stop_ts[p_sym] = datetime.now(timezone.utc)
                                self.broker.client.submit_order(StopOrderRequest(
                                    symbol=p_sym, qty=stop_qty,
                                    side=AlpacaSide.SELL if is_long else AlpacaSide.BUY,
                                    time_in_force=AlpacaTIF.GTC,
                                    stop_price=round(sl_level, 2),
                                ))
                                self._add_log(f"🛡️ STOP REAL colocado en Alpaca: {p_sym} x{stop_qty} @ ${sl_level:.2f}", "success")
                        except Exception as stop_err:
                            logger.error("broker_stop_placement_error", symbol=p_sym, error=str(stop_err))

                    # 4. SALIDA POR REVERSIÓN DE TENDENCIA, COMITÉ DE 10 AGENTES O SHOCK DE NOTICIAS/GEMINI:
                    if not close_reason:
                        pos_news_score = self.latest_news_scores.get(p_sym, 0.0)
                        if is_long and pos_news_score <= -45.0:
                            close_reason = f"📰 SALIDA POR NOTICIA ADVERSA / GEMINI: {p_sym} LONG cerrado por catalizador bajista severo (score {pos_news_score:+.1f})."
                        elif not is_long and pos_news_score >= 45.0:
                            close_reason = f"📰 SALIDA POR NOTICIA ALCISTA / GEMINI: {p_sym} SHORT cerrado por catalizador alcista severo (score {pos_news_score:+.1f})."

                    if not close_reason:
                        fv = self.scanner.feature_store.get(p_sym)
                        if fv:
                            st_dir = getattr(fv, "supertrend_dir", 0)
                            ema_spread = getattr(fv, "ema_9_21_spread", 0.0)
                            if is_long and st_dir == -1 and ema_spread < 0:
                                close_reason = f"🔄 REVERSIÓN TÉCNICA: {p_sym} LONG cerrado por cambio a tendencia bajista diaria (Supertrend -1)."
                            elif not is_long and st_dir == 1 and ema_spread > 0:
                                close_reason = f"🔄 REVERSIÓN TÉCNICA: {p_sym} SHORT cerrado por cambio a tendencia alcista diaria (Supertrend +1)."
                            else:
                                # Evaluar si el Comité de 10 Agentes giró fuertemente en dirección opuesta a la posición abierta
                                try:
                                    pos_f_dict = fv.to_dict()
                                    if hasattr(self.news_engine, "to_feature_dict"):
                                        pos_f_dict.update(self.news_engine.to_feature_dict(p_sym))
                                    pos_signals = await asyncio.gather(*(
                                        ag.analyze(p_sym, pos_f_dict, getattr(self.regime_manager, "current_regime", None))
                                        for ag in self.committee_agents
                                    ))
                                    valid_pos_sigs = [s for s in pos_signals if s is not None]
                                    pos_ens = self.signal_combiner.combine(
                                        p_sym, valid_pos_sigs, news_composite_score=self.latest_news_scores.get(p_sym, 0.0)
                                    )
                                    if pos_ens.is_approved:
                                        if is_long and pos_ens.direction.value == "sell":
                                            close_reason = f"🤖 SALIDA POR COMITÉ 10 AGENTES: {p_sym} LONG cerrado por consenso de VENTA ({len(pos_ens.contributing_strategies)} agentes, score {pos_ens.composite_score:.2f})."
                                        elif not is_long and pos_ens.direction.value == "buy":
                                            close_reason = f"🤖 SALIDA POR COMITÉ 10 AGENTES: {p_sym} SHORT cerrado por consenso de COMPRA ({len(pos_ens.contributing_strategies)} agentes, score {pos_ens.composite_score:.2f})."
                                except Exception:
                                    pass

                    # 5. SALIDA POR MADUREZ DE SWING / OPERACIÓN ESTANCADA:
                    # a) Si lleva 5 o más días de mercado y ya acumuló ganancia sólida (>= +1.5%), cerrar y asegurar beneficios!
                    # b) Si lleva 7 o más días sin avance (plana < +1.0%), liberar capital.
                    if not close_reason:
                        entry_iso = self.position_entry_times.get(p_sym)
                        if entry_iso:
                            try:
                                entry_dt = datetime.fromisoformat(entry_iso)
                                days_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400.0
                                if days_held >= 5.0 and gain_pct >= 0.015:
                                    close_reason = f"🎯 TOMA DE GANANCIAS POR MADUREZ ({days_held:.1f} días): {p_sym} acumuló +{gain_pct*100:.2f}%. Beneficio asegurado en cuenta."
                                elif days_held >= 7.0 and gain_pct < 0.010:
                                    close_reason = f"⏳ SALIDA POR ESTANCAMIENTO ({days_held:.1f} días): {p_sym} lateral sin alcanzar objetivo. Liberando capital."
                            except Exception:
                                pass

                    # 5.5. SALIDA PARCIAL en 1R: vender la mitad una sola vez y dejar correr el resto
                    if not close_reason:
                        from services.risk_engine.position_sizing import partial_exit_qty
                        part_qty = partial_exit_qty(
                            quantity=pos.quantity,
                            entry_price=entry,
                            mark_price=mark,
                            initial_stop=initial_sl,
                            is_long=is_long,
                            already_taken=self.position_partial_taken.get(p_sym, False),
                        )
                        last_exit = self._exit_cooldown.get(p_sym)
                        recently_sent = bool(last_exit and (datetime.now(timezone.utc) - last_exit).total_seconds() < 180)
                        if part_qty > 0 and not recently_sent:
                            self._exit_cooldown[p_sym] = datetime.now(timezone.utc)
                            try:
                                for oid in sym_to_order_ids.get(p_sym, []):
                                    try:
                                        self.broker.client.cancel_order_by_id(oid)
                                    except Exception:
                                        pass
                                part_side = AlpacaSide.SELL if is_long else AlpacaSide.BUY
                                if is_extended_session:
                                    from alpaca.trading.requests import LimitOrderRequest
                                    p_lim = round(mark * (0.995 if is_long else 1.005), 2)
                                    part_req = LimitOrderRequest(
                                        symbol=p_sym, qty=str(part_qty), side=part_side,
                                        time_in_force=AlpacaTIF.DAY, limit_price=p_lim, extended_hours=True,
                                    )
                                else:
                                    part_req = MarketOrderRequest(
                                        symbol=p_sym, qty=str(part_qty), side=part_side, time_in_force=AlpacaTIF.DAY,
                                    )
                                self.broker.client.submit_order(part_req)
                                self.position_partial_taken[p_sym] = True
                                self._save_worker_config()
                                self._add_log(
                                    f"💰 SALIDA PARCIAL: {p_sym} alcanzó 1R (+{gain_pct*100:.2f}%) · vendidas {part_qty} de {abs(int(pos.quantity))} acciones. El resto sigue con break-even/trailing.",
                                    "success",
                                )
                                logger.info("partial_exit_sent", symbol=p_sym, qty=part_qty, mark=mark, entry=entry)
                            except Exception as part_err:
                                logger.error("partial_exit_error", symbol=p_sym, error=str(part_err))
                            continue

                    if close_reason:
                        last_exit = self._exit_cooldown.get(p_sym)
                        if last_exit and (datetime.now(timezone.utc) - last_exit).total_seconds() < 180:
                            continue  # a close order was already sent recently; wait for the fill
                        self._exit_cooldown[p_sym] = datetime.now(timezone.utc)
                        self.position_peaks.pop(p_sym, None)
                        self.position_entry_times.pop(p_sym, None)
                        self.position_be_activated.pop(p_sym, None)
                        self.position_partial_taken.pop(p_sym, None)
                        self._save_worker_config()
                        logger.warning(
                            "software_exit_triggered",
                            symbol=p_sym,
                            side=pos.side,
                            entry=entry,
                            mark=mark,
                            reason=close_reason,
                        )
                        self._add_log(close_reason, "success" if ("TAKE-PROFIT" in close_reason or "TRANCA" in close_reason or "BREAK-EVEN" in close_reason) else "error")

                        # Cancel any existing open orders for this symbol to free shares
                        for oid in sym_to_order_ids.get(p_sym, []):
                            try:
                                self.broker.client.cancel_order_by_id(oid)
                            except Exception:
                                pass

                        # Submit close order immediately (Limit with extended_hours if outside regular hours)
                        qty_str = str(abs(int(pos.quantity)))
                        close_side = AlpacaSide.SELL if is_long else AlpacaSide.BUY
                        try:
                            if is_extended_session:
                                from alpaca.trading.requests import LimitOrderRequest
                                lim_p = round(mark * (0.995 if is_long else 1.005), 2)
                                mkt_close = LimitOrderRequest(
                                    symbol=p_sym,
                                    qty=qty_str,
                                    side=close_side,
                                    time_in_force=AlpacaTIF.DAY,
                                    limit_price=lim_p,
                                    extended_hours=True,
                                )
                            else:
                                mkt_close = MarketOrderRequest(
                                    symbol=p_sym,
                                    qty=qty_str,
                                    side=close_side,
                                    time_in_force=AlpacaTIF.GTC,
                                )
                            self.broker.client.submit_order(mkt_close)
                            logger.info("software_exit_close_sent", symbol=p_sym, extended=is_extended_session, reason=close_reason)
                            self._add_log(f"⚡ Orden de cierre enviada para {p_sym} ({qty_str} acciones)", "info")
                        except Exception as exit_err:
                            logger.error("software_exit_close_error", symbol=p_sym, error=str(exit_err))
        except Exception as sl_monitor_err:
            logger.warning("software_sl_tp_monitor_error", error=str(sl_monitor_err))

        # 7.8. Shadow Mode Position Lifecycle Evaluation
        if self.shadow_mode and hasattr(self, "shadow_engine"):
            current_prices = {
                sym: float(getattr(fv, "close_price", 0.0))
                for sym, fv in self.scanner.feature_store.items()
                if hasattr(fv, "close_price") and getattr(fv, "close_price", 0.0) > 0
            }
            closed_shadow_trades = self.shadow_engine.evaluate_open_positions(current_prices)
            for strd in closed_shadow_trades:
                self._add_log(
                    f"🧪 [SHADOW EXIT] {strd.symbol} cerrado @ ${strd.exit_price:.2f} ({strd.exit_reason}) · Net P&L: ${strd.net_pnl:.2f}",
                    "success" if strd.net_pnl > 0 else "warning",
                )

        # 8. Reconcile broker positions and mark-to-market
        await self.broker.reconcile()

        # 8.1. Harness² Real-Time Closed Trade Reflection & Diagnostics
        try:
            current_positions = await self.broker.get_positions()
            current_syms = {p.symbol: p for p in current_positions}
            if hasattr(self, "_tracked_open_positions") and self._tracked_open_positions:
                for prev_sym, prev_pos in list(self._tracked_open_positions.items()):
                    if prev_sym not in current_syms:
                        entry_px = float(getattr(prev_pos, "avg_entry_price", 0.0))
                        fv = self.scanner.feature_store.get(prev_sym) if hasattr(self, "scanner") else None
                        cur_px = float(getattr(fv, "close_price", entry_px)) if fv else entry_px
                        qty = float(getattr(prev_pos, "quantity", 1.0))
                        is_long = getattr(prev_pos, "side", "").lower() == "long"
                        pnl_usd = (cur_px - entry_px) * qty if is_long else (entry_px - cur_px) * abs(qty)
                        pnl_pct = ((cur_px - entry_px) / entry_px * 100.0) if (is_long and entry_px > 0) else (((entry_px - cur_px) / entry_px * 100.0) if entry_px > 0 else 0.0)
                        diag = self.harness_engine.record_closed_trade(
                            symbol=prev_sym,
                            side="BUY" if is_long else "SELL",
                            entry_price=entry_px,
                            exit_price=cur_px,
                            quantity=qty,
                            pnl_usd=pnl_usd,
                            pnl_pct=pnl_pct,
                        )
                        self._add_log(
                            f"🧠 [HARNESS² REFLEXIÓN] {prev_sym} cerrado ({diag.failure_category.value}): PnL ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)",
                            "success" if pnl_usd >= 0 else "warning",
                        )
            self._tracked_open_positions = current_syms
        except Exception as harness_err:
            logger.warning("harness_reflection_cycle_error", error=str(harness_err))

        snapshot = self.get_telemetry_snapshot()
        return snapshot

    async def start(self):
        """Starts continuous autonomous trading loop."""
        self.is_running = True
        logger.info("quant_trading_worker_started", interval=self.scan_interval_seconds)

        # Allow API server to complete boot sequence and accept requests first
        await asyncio.sleep(1.5)

        while self.is_running:
            try:
                await self.execute_cycle()
            except Exception as exc:
                logger.error("quant_worker_cycle_error", error=str(exc))

            caps = get_session_capabilities()
            if caps.granular_session in {"weekend_analysis", "overnight_analysis", "holiday_analysis"}:
                # On weekend or overnight, sleep at dynamic cadence (max 300s to remain responsive to commands)
                sleep_sec = min(300.0, caps.recommended_cycle_seconds)
            else:
                sleep_sec = self.scan_interval_seconds

            await asyncio.sleep(sleep_sec)


    def stop(self):
        """Stops the trading loop gracefully."""
        self.is_running = False
        logger.info("quant_trading_worker_stopped")


_global_worker: Optional[QuantTradingWorker] = None


def get_global_worker() -> QuantTradingWorker:
    """Returns the shared singleton QuantTradingWorker instance."""
    global _global_worker
    if _global_worker is None:
        _global_worker = QuantTradingWorker()
    return _global_worker


if __name__ == "__main__":
    import asyncio
    print("\n[+] Iniciando Quant Trading Daemon directamente...")
    worker = get_global_worker()
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        worker.stop()
        print("\n[i] Quant Trading Daemon detenido.")