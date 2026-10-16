"""AI Quant Trading Platform — Alpaca Broker Implementation.

Connects the platform directly to Alpaca Markets (Paper or Live).
Implements BrokerInterface:
- Live Account Balance & Buying Power via /v2/account
- Bracket Orders (Entry Market/Limit + Take-Profit Limit + Stop-Loss Stop) via /v2/orders
- Real-time Position tracking via /v2/positions
- Open & Closed Order audit trail via /v2/orders
- Emergency Kill Switch integration (cancel_all_orders)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
import asyncio
import uuid

from core.config import get_settings
from core.exceptions import InsufficientFundsError, OrderRejectedError, BrokerStateUnknownError
from core.logging import get_logger
from services.broker_engine.base import (
    AccountInfo,
    BrokerInterface,
    OrderClass,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        TakeProfitRequest,
        StopLossRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide as AlpacaOrderSide,
        TimeInForce as AlpacaTimeInForce,
        OrderClass as AlpacaOrderClass,
        QueryOrderStatus,
        OrderStatus as AlpacaOrderStatus,
    )
    from alpaca.common.exceptions import APIError
    _HAS_ALPACA_SDK = True
except ImportError:
    TradingClient = None  # type: ignore
    MarketOrderRequest = LimitOrderRequest = StopOrderRequest = TakeProfitRequest = StopLossRequest = GetOrdersRequest = None  # type: ignore
    AlpacaOrderSide = AlpacaTimeInForce = AlpacaOrderClass = QueryOrderStatus = AlpacaOrderStatus = None  # type: ignore
    APIError = Exception  # type: ignore
    _HAS_ALPACA_SDK = False

logger = get_logger("alpaca_broker")

ALPACA_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "held": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "replaced": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.CANCEL_PENDING,
    "pending_replace": OrderStatus.CANCEL_PENDING,
    "stopped": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "calculated": OrderStatus.SUBMITTED,
}


class AlpacaBroker(BrokerInterface):
    """
    Direct connector to Alpaca Markets Trading API.
    Works transparently in both Paper and Live mode.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        is_paper: Optional[bool] = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.alpaca.api_key
        raw_secret = secret_key or str(settings.alpaca.secret_key)
        self.secret_key = raw_secret
        self._is_paper = is_paper if is_paper is not None else settings.alpaca.is_paper

        self.client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self._is_paper,
        ) if TradingClient is not None else None
        self._connected: bool = False
        self.cash: float = 100000.0
        self.equity: float = 100000.0
        self.buying_power: float = 400000.0
        self.initial_cash: float = 100000.0
        self.last_equity: float = 100000.0
        self.realized_pnl: float = 0.0
        self.realized_pnl_today: float = 0.0
        self._history_cache: list[dict[str, Any]] = []
        self._history_cache_ts: float = 0.0
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}
        self._order_requests: dict[str, OrderRequest] = {}
        self._order_metadata: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "alpaca_paper" if self._is_paper else "alpaca_live"

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    # ── Lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        if getattr(self, "_connect_attempted", False) and not self._connected:
            return

        self._connect_attempted = True
        try:
            account = await asyncio.wait_for(
                asyncio.to_thread(self.client.get_account),
                timeout=2.5,
            )
            self._connected = True
            self.cash = float(account.cash)
            self.equity = float(account.equity)
            self.buying_power = float(account.buying_power)
            self.last_equity = float(getattr(account, "last_equity", None) or self.equity)
            self.initial_cash = 100000.0
            await self.get_positions()
            await self.get_open_orders()
            logger.info(
                "alpaca_broker_connected",
                account_id=str(account.id),
                status=str(account.status),
                cash=self.cash,
                equity=self.equity,
                is_paper=self._is_paper,
            )
        except Exception as exc:
            self._connected = False
            logger.warning("alpaca_broker_offline_using_paper_mode", error=str(exc))

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("alpaca_broker_disconnected")

    # ── Account ──────────────────────────────────────────────────────

    async def get_account(self) -> AccountInfo:
        try:
            acc = self.client.get_account()
            self.equity = float(acc.equity)
            self.cash = float(acc.cash)
            self.buying_power = float(acc.buying_power)
            self.last_equity = float(getattr(acc, "last_equity", None) or self.equity)
            pv = float(acc.portfolio_value or acc.equity)
            return AccountInfo(
                account_id=str(acc.id),
                equity=round(self.equity, 2),
                cash=round(self.cash, 2),
                buying_power=round(self.buying_power, 2),
                portfolio_value=round(pv, 2),
                trading_mode="paper" if self._is_paper else "live",
            )
        except Exception as exc:
            logger.error("alpaca_get_account_error", error=str(exc))
            raise BrokerStateUnknownError(f"Cannot retrieve Alpaca account: {exc}")

    # ── Orders ───────────────────────────────────────────────────────

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        if not self._connected:
            await self.connect()

        is_buy = order.side == OrderSide.BUY
        alpaca_side = AlpacaOrderSide.BUY if is_buy else AlpacaOrderSide.SELL
        qty = round(order.quantity, 0)
        if qty <= 0:
            qty = 1.0

        is_bracket = bool(order.order_class == OrderClass.BRACKET and order.take_profit_price and order.stop_loss_price)

        try:
            order_type_submitted = order.order_type
            if order.extended_hours:
                # Horario Extendido (Pre-Market 4:00-9:30 ET / After-Hours 16:00-20:00 ET)
                # Para este sistema, las órdenes extended-hours se fuerzan a DAY como política de ejecución
                # y control de riesgo, aunque Alpaca actualmente admite DAY y GTC para órdenes limitadas extended-hours.
                limit_p = order.limit_price or (order.take_profit_price if not is_buy else order.stop_loss_price)
                if not limit_p or limit_p <= 0.0:
                    raise ValueError(f"Extended hours orders require an explicit limit_price for symbol {order.symbol}; synthetic fallback is forbidden.")
                limit_p = round(limit_p, 2)
                req_kwargs: dict[str, Any] = {
                    "symbol": order.symbol,
                    "qty": qty,
                    "side": alpaca_side,
                    "limit_price": limit_p,
                    "time_in_force": AlpacaTimeInForce.DAY,
                    "extended_hours": True,
                }
                if order.client_order_id:
                    req_kwargs["client_order_id"] = order.client_order_id
                req = LimitOrderRequest(**req_kwargs)
                order_type_submitted = OrderType.LIMIT
            elif is_bracket:
                tp_req = TakeProfitRequest(limit_price=round(order.take_profit_price, 2))
                sl_req = StopLossRequest(stop_price=round(order.stop_loss_price, 2))

                req_kwargs = {
                    "symbol": order.symbol,
                    "qty": qty,
                    "side": alpaca_side,
                    "time_in_force": AlpacaTimeInForce.GTC,
                    "order_class": AlpacaOrderClass.BRACKET,
                    "take_profit": tp_req,
                    "stop_loss": sl_req,
                }
                if order.client_order_id:
                    req_kwargs["client_order_id"] = order.client_order_id
                req = MarketOrderRequest(**req_kwargs)
            elif order.order_type == OrderType.LIMIT and order.limit_price is not None:
                req_kwargs = {
                    "symbol": order.symbol,
                    "qty": qty,
                    "side": alpaca_side,
                    "limit_price": round(order.limit_price, 2),
                    "time_in_force": AlpacaTimeInForce.DAY,
                }
                if order.client_order_id:
                    req_kwargs["client_order_id"] = order.client_order_id
                req = LimitOrderRequest(**req_kwargs)
            else:
                req_kwargs = {
                    "symbol": order.symbol,
                    "qty": qty,
                    "side": alpaca_side,
                    "time_in_force": AlpacaTimeInForce.DAY,
                }
                if order.client_order_id:
                    req_kwargs["client_order_id"] = order.client_order_id
                req = MarketOrderRequest(**req_kwargs)

            alp_res = self.client.submit_order(req)
            order_id = str(alp_res.id)
            self._order_requests[order_id] = order

            status_str = str(alp_res.status.value if hasattr(alp_res.status, "value") else alp_res.status).lower()
            mapped_status = ALPACA_STATUS_MAP.get(status_str, OrderStatus.SUBMITTED)

            res = OrderResult(
                broker_order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order_type_submitted,
                status=mapped_status,
                quantity=qty,
                filled_qty=float(alp_res.filled_qty or 0.0),
                filled_avg_price=float(alp_res.filled_avg_price) if alp_res.filled_avg_price else None,
                submitted_at=alp_res.submitted_at or datetime.now(timezone.utc),
                raw_response={
                    "id": order_id,
                    "raw_status": status_str,
                    "type_display": "Limit" if order_type_submitted == OrderType.LIMIT else "Market",
                    "client_order_id": order.client_order_id,
                    "legs": [str(l.id) for l in alp_res.legs] if getattr(alp_res, "legs", None) else [],
                },
            )

            logger.info(
                "alpaca_order_submitted",
                order_id=order_id,
                symbol=order.symbol,
                side=order.side.value,
                qty=qty,
                is_bracket=is_bracket,
                status=mapped_status.value,
            )
            return res

        except APIError as api_err:
            if order.client_order_id:
                try:
                    existing = await self.get_order_by_client_order_id(order.client_order_id)
                    if existing:
                        logger.warning("alpaca_recovered_order_after_api_error", client_order_id=order.client_order_id)
                        return existing
                except Exception:
                    pass
            logger.error("alpaca_api_error_submitting_order", symbol=order.symbol, error=str(api_err))
            raise OrderRejectedError(f"Alpaca rejected order: {api_err}", order_id=order.symbol)
        except Exception as exc:
            if order.client_order_id:
                try:
                    existing = await self.get_order_by_client_order_id(order.client_order_id)
                    if existing:
                        logger.warning("alpaca_recovered_order_after_network_error", client_order_id=order.client_order_id)
                        return existing
                except Exception:
                    pass
            logger.error("alpaca_unexpected_error_submitting_order", symbol=order.symbol, error=str(exc))
            raise OrderRejectedError(f"Failed to submit to Alpaca: {exc}", order_id=order.symbol)

    async def cancel_order(self, broker_order_id: str) -> OrderResult:
        try:
            self.client.cancel_order_by_id(uuid.UUID(broker_order_id))
            logger.info("alpaca_order_cancelled", order_id=broker_order_id)
            return OrderResult(
                broker_order_id=broker_order_id,
                symbol="UNKNOWN",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.CANCELLED,
                quantity=0.0,
            )
        except Exception as exc:
            logger.warning("alpaca_cancel_order_error", order_id=broker_order_id, error=str(exc))
            raise OrderRejectedError(f"Could not cancel Alpaca order {broker_order_id}: {exc}")

    async def cancel_all_orders(self) -> list[OrderResult]:
        try:
            self.client.cancel_orders()
            logger.info("alpaca_all_orders_cancelled")
            return []
        except Exception as exc:
            logger.error("alpaca_cancel_all_orders_error", error=str(exc))
            return []

    async def get_order(self, broker_order_id: str) -> OrderResult:
        alp_ord = self.client.get_order_by_id(uuid.UUID(broker_order_id))
        status_str = str(alp_ord.status.value if hasattr(alp_ord.status, "value") else alp_ord.status).lower()
        mapped_status = ALPACA_STATUS_MAP.get(status_str, OrderStatus.SUBMITTED)
        side = OrderSide.BUY if str(alp_ord.side).lower() == "buy" else OrderSide.SELL
        return OrderResult(
            broker_order_id=str(alp_ord.id),
            symbol=alp_ord.symbol,
            side=side,
            order_type=OrderType.MARKET,
            status=mapped_status,
            quantity=float(alp_ord.qty),
            filled_qty=float(alp_ord.filled_qty or 0.0),
            filled_avg_price=float(alp_ord.filled_avg_price) if alp_ord.filled_avg_price else None,
            submitted_at=alp_ord.submitted_at,
            filled_at=alp_ord.filled_at,
        )

    async def get_order_by_client_order_id(self, client_order_id: str) -> OrderResult | None:
        """Query Alpaca REST API for order with specific client_order_id."""
        try:
            alp_ord = self.client.get_order_by_client_order_id(client_order_id)
            if not alp_ord:
                return None
            status_str = str(alp_ord.status.value if hasattr(alp_ord.status, "value") else alp_ord.status).lower()
            mapped_status = ALPACA_STATUS_MAP.get(status_str, OrderStatus.SUBMITTED)
            side = OrderSide.BUY if str(alp_ord.side).lower() == "buy" else OrderSide.SELL
            return OrderResult(
                broker_order_id=str(alp_ord.id),
                symbol=alp_ord.symbol,
                side=side,
                order_type=OrderType.LIMIT if getattr(alp_ord, "limit_price", None) else OrderType.MARKET,
                status=mapped_status,
                quantity=float(alp_ord.qty or 0.0),
                filled_qty=float(alp_ord.filled_qty or 0.0),
                filled_avg_price=float(alp_ord.filled_avg_price) if alp_ord.filled_avg_price else None,
                submitted_at=alp_ord.submitted_at,
                filled_at=alp_ord.filled_at,
                raw_response={"client_order_id": client_order_id},
            )
        except Exception as exc:
            logger.debug("alpaca_get_order_by_client_order_id_miss", client_order_id=client_order_id, error=str(exc))
            return None

    async def get_open_orders(self) -> list[OrderResult]:
        if not self._connected:
            raise BrokerStateUnknownError("Alpaca broker is disconnected; open orders state unknown.")
        try:
            orders = await asyncio.wait_for(
                asyncio.to_thread(self.client.get_orders, GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)),
                timeout=2.0,
            )
            results = []
            for o in orders:
                status_str = str(o.status.value if hasattr(o.status, "value") else o.status).lower()
                mapped_status = ALPACA_STATUS_MAP.get(status_str, OrderStatus.SUBMITTED)
                side = OrderSide.BUY if str(o.side.value if hasattr(o.side, "value") else o.side).lower() == "buy" else OrderSide.SELL
                o_type_str = str(o.order_type.value if hasattr(o.order_type, "value") else o.order_type).lower()
                order_type = OrderType.LIMIT if o_type_str == "limit" else (OrderType.STOP if o_type_str == "stop" else OrderType.MARKET)

                limit_p = float(o.limit_price) if getattr(o, "limit_price", None) is not None else None
                stop_p = float(o.stop_price) if getattr(o, "stop_price", None) is not None else None
                
                type_display = "Market"
                if order_type == OrderType.LIMIT and limit_p:
                    type_display = f"Limit @ ${limit_p:.2f}"
                elif order_type == OrderType.STOP and stop_p:
                    type_display = f"Stop @ ${stop_p:.2f}"

                results.append(OrderResult(
                    broker_order_id=str(o.id),
                    symbol=o.symbol,
                    side=side,
                    order_type=order_type,
                    status=mapped_status,
                    quantity=float(o.qty),
                    filled_qty=float(o.filled_qty or 0.0),
                    filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                    submitted_at=o.submitted_at,
                    raw_response={
                        "raw_status": status_str,
                        "type_display": type_display,
                        "limit_price": limit_p,
                        "stop_price": stop_p,
                    }
                ))

                # Include bracket child legs (Stop Loss & Take Profit)
                if getattr(o, "legs", None):
                    for leg in o.legs:
                        leg_status_str = str(leg.status.value if hasattr(leg.status, "value") else leg.status).lower()
                        leg_mapped_status = ALPACA_STATUS_MAP.get(leg_status_str, OrderStatus.SUBMITTED)
                        leg_side = OrderSide.BUY if str(leg.side.value if hasattr(leg.side, "value") else leg.side).lower() == "buy" else OrderSide.SELL
                        leg_type_str = str(leg.order_type.value if hasattr(leg.order_type, "value") else leg.order_type).lower()
                        leg_order_type = OrderType.LIMIT if leg_type_str == "limit" else (OrderType.STOP if leg_type_str == "stop" else OrderType.MARKET)
                        leg_limit_p = float(leg.limit_price) if getattr(leg, "limit_price", None) is not None else None
                        leg_stop_p = float(leg.stop_price) if getattr(leg, "stop_price", None) is not None else None

                        leg_display = "Market"
                        if leg_order_type == OrderType.LIMIT and leg_limit_p:
                            leg_display = f"Limit @ ${leg_limit_p:.2f}"
                        elif leg_order_type == OrderType.STOP and leg_stop_p:
                            leg_display = f"Stop @ ${leg_stop_p:.2f}"

                        results.append(OrderResult(
                            broker_order_id=str(leg.id),
                            symbol=leg.symbol,
                            side=leg_side,
                            order_type=leg_order_type,
                            status=leg_mapped_status,
                            quantity=float(leg.qty),
                            filled_qty=float(leg.filled_qty or 0.0),
                            filled_avg_price=float(leg.filled_avg_price) if leg.filled_avg_price else None,
                            submitted_at=leg.submitted_at,
                            raw_response={
                                "raw_status": leg_status_str,
                                "type_display": leg_display,
                                "limit_price": leg_limit_p,
                                "stop_price": leg_stop_p,
                            }
                        ))

            self._orders = {r.broker_order_id: r for r in results}
            return results
        except Exception as exc:
            logger.error("alpaca_get_open_orders_error", error=str(exc))
            raise BrokerStateUnknownError(f"Cannot retrieve Alpaca open orders: {exc}")

    # ── Positions ────────────────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        if not self._connected:
            raise BrokerStateUnknownError("Alpaca broker is disconnected; positions state unknown.")
        try:
            alp_pos = await asyncio.wait_for(
                asyncio.to_thread(self.client.get_all_positions),
                timeout=2.0,
            )
            results = []
            for p in alp_pos:
                side = str(p.side).lower()
                results.append(Position(
                    symbol=p.symbol,
                    side="long" if side in ["long", "positionside.long"] else "short",
                    quantity=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pnl=float(p.unrealized_pl or 0.0),
                    unrealized_pnl_pct=float(p.unrealized_plpc or 0.0) * 100.0,
                ))
            self._positions = {p.symbol: p for p in results}
            return results
        except Exception as exc:
            logger.error("alpaca_get_positions_error", error=str(exc))
            raise BrokerStateUnknownError(f"Cannot retrieve Alpaca positions: {exc}")

    async def get_position(self, symbol: str) -> Position | None:
        if not self._connected:
            return self._positions.get(symbol)
        try:
            p = self.client.get_open_position(symbol)
            side = str(p.side).lower()
            return Position(
                symbol=p.symbol,
                side="long" if side in ["long", "positionside.long"] else "short",
                quantity=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl or 0.0),
                unrealized_pnl_pct=float(p.unrealized_plpc or 0.0) * 100.0,
            )
        except Exception:
            return None

    # ── Trade History & Realized P&L ─────────────────────────────────

    def get_trade_history(self) -> list[dict[str, Any]]:
        """Build closed round trips from filled orders using FIFO matching.

        Looks back 30 days (up to 500 closed orders, mostly cancelled bracket legs are
        skipped) and refreshes realized_pnl / realized_pnl_today as a side effect.
        Results are cached for 20s so telemetry polling does not hammer the API.
        """
        import time as _time
        from datetime import timedelta

        if not self._connected:
            return []
        if self._history_cache_ts and (_time.time() - self._history_cache_ts) < 20.0:
            return self._history_cache
        try:
            closed_orders = self.client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=500,
                after=datetime.now(timezone.utc) - timedelta(days=30),
                nested=False,
            ))
            fills = []
            for o in closed_orders:
                status_val = str(getattr(o.status, "value", o.status)).lower()
                if status_val not in ["filled", "orderstatus.filled"]:
                    continue
                side_str = "BUY" if str(getattr(o.side, "value", o.side)).lower().endswith("buy") else "SELL"
                qty = float(o.filled_qty or o.qty or 0.0)
                filled_p = float(o.filled_avg_price or 0.0)
                if qty <= 0 or filled_p <= 0:
                    continue
                event_time = o.filled_at or o.submitted_at
                fills.append({
                    "id": str(o.id), "symbol": o.symbol, "side": side_str,
                    "qty": qty, "price": filled_p,
                    "time": event_time,
                })

            fills.sort(key=lambda item: item["time"] or datetime.min.replace(tzinfo=timezone.utc))
            lots: dict[str, list[dict[str, Any]]] = {}
            trades: list[dict[str, Any]] = []
            for fill in fills:
                symbol = fill["symbol"]
                opposite = "SELL" if fill["side"] == "BUY" else "BUY"
                book = lots.setdefault(symbol, [])
                remaining = fill["qty"]
                while remaining > 1e-9 and book and book[0]["side"] == opposite:
                    lot = book[0]
                    matched = min(remaining, lot["qty"])
                    is_long = lot["side"] == "BUY"
                    pnl = (fill["price"] - lot["price"]) * matched if is_long else (lot["price"] - fill["price"]) * matched
                    denom = lot["price"] * matched
                    trades.append({
                        "trade_id": f"alp_{fill['id'][:8]}_{lot['id'][:8]}",
                        "order_id": fill["id"][:12],
                        "symbol": symbol,
                        "side": "LONG" if is_long else "SHORT",
                        "qty": matched,
                        "entry_price": lot["price"],
                        "exit_price": fill["price"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round((pnl / denom) * 100.0, 4) if denom else 0.0,
                        "outcome": "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN"),
                        "reason": "ALPACA_FILLED_CLOSE",
                        "closed_at": fill["time"].isoformat() if fill["time"] else "",
                    })
                    lot["qty"] -= matched
                    remaining -= matched
                    if lot["qty"] <= 1e-9:
                        book.pop(0)
                if remaining > 1e-9:
                    book.append({"id": fill["id"], "side": fill["side"], "qty": remaining, "price": fill["price"]})

            result = list(reversed(trades))
            self.realized_pnl = round(sum(t["pnl"] for t in trades), 2)
            today = datetime.now(timezone.utc).date().isoformat()
            self.realized_pnl_today = round(
                sum(t["pnl"] for t in trades if str(t.get("closed_at", "")).startswith(today)), 2
            )
            self._history_cache = result
            self._history_cache_ts = _time.time()
            return result
        except Exception as exc:
            logger.warning("alpaca_get_trade_history_error", error=str(exc))
            return self._history_cache

    # ── Reconciliation & Health ──────────────────────────────────────

    async def reconcile(self) -> dict[str, Any]:
        """
        Verify internal consistency against Alpaca based on Broker Truth.
        Ensures execution prices are valid and handles missing prices safely.
        """
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                pass
        if not self._connected:
            return {
                "status": "disconnected",
                "account_equity": getattr(self, "equity", 100000.0),
                "cash": getattr(self, "cash", 100000.0),
                "open_positions": len(self._positions),
                "open_orders": len(self._orders),
                "discrepancies": ["Alpaca broker offline / disconnected"],
                "pending_reconciliations": [],
                "provider": "alpaca_official",
            }

        acc = await self.get_account()
        pos = await self.get_positions()
        orders = await self.get_open_orders()
        discrepancies = []
        pending_reconciliations = []

        # Broker Truth: Audit local tracking against broker reality
        broker_positions_map = {p.symbol: p for p in pos}
        for sym, local_p in list(self._positions.items()):
            if sym not in broker_positions_map:
                discrepancies.append(
                    f"CRITICAL DISCREPANCY: Local state tracked {local_p.quantity} shares of {sym}, but broker reports 0. Synced to broker truth."
                )
                del self._positions[sym]
            elif broker_positions_map[sym].quantity != local_p.quantity:
                discrepancies.append(
                    f"DISCREPANCY: Quantity mismatch for {sym}: local={local_p.quantity}, broker={broker_positions_map[sym].quantity}. Synced to broker truth."
                )
                self._positions[sym] = broker_positions_map[sym]

        for sym, b_pos in broker_positions_map.items():
            if sym not in self._positions:
                discrepancies.append(
                    f"DISCREPANCY: Broker reports open position for {sym} ({b_pos.quantity} shares @ ${b_pos.avg_entry_price:.2f}) not in local tracking. Synced to broker truth."
                )
                self._positions[sym] = b_pos

        if self._connected:
            try:
                # Query recent closed orders to verify fill prices
                closed_orders = await asyncio.to_thread(
                    self.client.get_orders,
                    GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=20)
                )
                for o in closed_orders:
                    st = str(o.status).lower()
                    if st in ["filled", "orderstatus.filled"]:
                        fill_p = float(o.filled_avg_price) if o.filled_avg_price else None
                        if fill_p is None or fill_p <= 0.0:
                            # Attempt re-query directly by order ID from broker
                            try:
                                refreshed = await asyncio.to_thread(self.client.get_order_by_id, o.id)
                                fill_p = float(refreshed.filled_avg_price) if refreshed.filled_avg_price else None
                            except Exception:
                                fill_p = None

                        if fill_p is None or fill_p <= 0.0:
                            # Execution price is unknown; flag as RECONCILIATION_PENDING. Never invent synthetic price ($100.0).
                            msg = (
                                f"RECONCILIATION_PENDING: Filled order {o.id} for {o.symbol} has unknown filled_avg_price. "
                                f"Accounting reconciliation halted for this order."
                            )
                            logger.error("alpaca_reconcile_missing_price", order_id=str(o.id), symbol=o.symbol)
                            pending_reconciliations.append({
                                "order_id": str(o.id),
                                "symbol": o.symbol,
                                "status": "RECONCILIATION_PENDING",
                                "reason": "UNKNOWN_EXECUTION_PRICE",
                            })
                            discrepancies.append(msg)
            except Exception as exc:
                logger.warning("alpaca_reconcile_orders_fetch_warning", error=str(exc))
                discrepancies.append(f"Could not verify recent closed orders: {exc}")

        status = "clean"
        if pending_reconciliations:
            status = "RECONCILIATION_PENDING"
        elif discrepancies:
            status = "reconciled_with_broker"

        return {
            "status": status,
            "account_equity": acc.equity,
            "cash": acc.cash,
            "open_positions": len(pos),
            "open_orders": len(orders),
            "discrepancies": discrepancies,
            "pending_reconciliations": pending_reconciliations,
            "provider": "alpaca_official",
        }

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "healthy" if self._connected else "disconnected",
            "provider": "alpaca_official",
            "mode": "paper" if self._is_paper else "live",
        }
