"""
AI Quant Trading Platform — Paper Broker Implementation.

High-fidelity simulated broker implementing BrokerInterface.
Features:
- Instant or realistic latency fills for MARKET orders
- Continuous limit & stop order matching on incoming price ticks/bars
- Native BRACKET (OCO) support: Take-Profit fills automatically cancel Stop-Loss
- Integration with CostModel for realistic execution friction (spread + slippage)
- Position mark-to-market tracking and buying power enforcement
- Full state transitions via OrderStateMachine
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.logging import get_logger
from core.exceptions import InsufficientFundsError, OrderRejectedError
from services.backtesting.cost_model import CostModel
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
from services.broker_engine.order_state_machine import OrderStateMachine

logger = get_logger("paper_broker")


class PaperBroker(BrokerInterface):
    """
    In-memory paper trading broker engine.
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        cost_model: CostModel | None = None,
        state_file: Path | str | None = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.cost_model = cost_model or CostModel()

        self._connected: bool = False
        self._orders: dict[str, OrderResult] = {}
        self._order_requests: dict[str, OrderRequest] = {}
        self._positions: dict[str, Position] = {}
        self._market_prices: dict[str, float] = {}

        # OCO linkage: parent_order_id -> [child_order_id_1, child_order_id_2]
        self._bracket_children: dict[str, list[str]] = {}
        # Sibling linkage: child_order_id -> sibling_order_id (for OCO cancellation)
        self._oco_siblings: dict[str, str] = {}

        # Trade History & Realized PnL
        self.realized_pnl: float = 0.0
        self._trade_history: list[dict[str, Any]] = []
        self._reconciled_order_ids: set[str] = set()

        # Persistence is opt-in so independent simulations cannot contaminate
        # one another. The long-running worker passes its account state file.
        self.state_file: Path | None = Path(state_file) if state_file else None
        if self.state_file is not None:
            self._load_persisted_state()

    def _save_persisted_state(self) -> None:
        """Persists current account balance, positions, and trade history to disk."""
        if self.state_file is None:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "cash": round(self.cash, 2),
                "initial_cash": round(self.initial_cash, 2),
                "realized_pnl": round(self.realized_pnl, 2),
                "trade_history": self._trade_history,
                "positions": {
                    sym: pos.model_dump(mode="json")
                    for sym, pos in self._positions.items()
                },
                "orders": {
                    oid: ord_res.model_dump(mode="json")
                    for oid, ord_res in self._orders.items()
                },
                "order_requests": {
                    oid: req.model_dump(mode="json")
                    for oid, req in self._order_requests.items()
                },
                "bracket_children": self._bracket_children,
                "oco_siblings": self._oco_siblings,
                "market_prices": self._market_prices,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            tmp = self.state_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            tmp.replace(self.state_file)
        except Exception as exc:
            logger.warning("save_paper_state_error", error=str(exc))

    def _load_persisted_state(self) -> None:
        """Restores account balance, closed trades, and active positions from disk if available."""
        if self.state_file is None or not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            persisted_initial_cash = float(data.get("initial_cash", self.initial_cash))
            if persisted_initial_cash != float(self.initial_cash):
                logger.info(
                    "paper_state_ignored_for_different_initial_cash",
                    persisted_initial_cash=persisted_initial_cash,
                    requested_initial_cash=self.initial_cash,
                )
                return
            self.cash = float(data.get("cash", self.initial_cash))
            self.initial_cash = persisted_initial_cash
            self.realized_pnl = float(data.get("realized_pnl", 0.0))
            self._trade_history = data.get("trade_history", [])
            self._bracket_children = data.get("bracket_children", {})
            self._oco_siblings = data.get("oco_siblings", {})
            self._market_prices = data.get("market_prices", {})

            # Restore positions
            raw_pos = data.get("positions", {})
            self._positions = {}
            for sym, p_data in raw_pos.items():
                try:
                    self._positions[sym] = Position(**p_data)
                except Exception:
                    pass

            # Restore orders
            raw_orders = data.get("orders", {})
            self._orders = {}
            for oid, o_data in raw_orders.items():
                try:
                    self._orders[oid] = OrderResult(**o_data)
                except Exception:
                    pass

            # Restore order requests
            raw_reqs = data.get("order_requests", {})
            self._order_requests = {}
            for oid, r_data in raw_reqs.items():
                try:
                    self._order_requests[oid] = OrderRequest(**r_data)
                except Exception:
                    pass

            logger.info(
                "paper_broker_state_restored",
                cash=self.cash,
                realized_pnl=self.realized_pnl,
                trades_count=len(self._trade_history),
                positions_count=len(self._positions),
            )
        except Exception as exc:
            logger.warning("load_paper_state_error", error=str(exc))


    @property
    def name(self) -> str:
        return "paper_broker"

    @property
    def is_paper(self) -> bool:
        return True

    # ── Lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._connected = True
        logger.info("paper_broker_connected", cash=self.cash)

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("paper_broker_disconnected")

    # ── Market Price Feeds for Matching ──────────────────────────────

    def update_market_price(
        self,
        symbol: str,
        price: float,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        """
        Update the latest price for a symbol and process pending limit/stop orders.
        """
        self._market_prices[symbol] = price
        h = high if high is not None else price
        l = low if low is not None else price

        # Update position mark-to-market
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = price
            pos.market_value = round(pos.quantity * price, 2)
            if pos.side == "long":
                pos.unrealized_pnl = round((price - pos.avg_entry_price) * pos.quantity, 2)
                pos.unrealized_pnl_pct = round(((price - pos.avg_entry_price) / pos.avg_entry_price) * 100.0, 2)
            else:
                pos.unrealized_pnl = round((pos.avg_entry_price - price) * pos.quantity, 2)
                pos.unrealized_pnl_pct = round(((pos.avg_entry_price - price) / pos.avg_entry_price) * 100.0, 2)

        # Match pending orders
        self._match_pending_orders(symbol, price, h, l)

    # ── Account ──────────────────────────────────────────────────────

    async def get_account(self) -> AccountInfo:
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        portfolio_val = sum(p.market_value if p.side.lower() == "long" else -p.market_value for p in self._positions.values())
        equity = self.cash + portfolio_val

        # Buying power (2x margin for paper broker)
        buying_power = max(0.0, self.cash * 2.0)

        return AccountInfo(
            account_id="PAPER_ACCOUNT_001",
            equity=round(equity, 2),
            cash=round(self.cash, 2),
            buying_power=round(buying_power, 2),
            portfolio_value=round(portfolio_val, 2),
            trading_mode="paper",
        )

    # ── Orders ───────────────────────────────────────────────────────

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        if not self._connected:
            raise RuntimeError("PaperBroker is not connected. Call connect() first.")

        order_id = f"paper_ord_{uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Record request
        self._order_requests[order_id] = order

        # Determine reference price
        ref_price = self._market_prices.get(order.symbol, order.limit_price or 100.0)

        # Buying power / cash validation
        is_buy = order.side == OrderSide.BUY
        est_cost = order.quantity * ref_price

        if is_buy and self.cash < est_cost:
            raise InsufficientFundsError(
                f"Insufficient cash for order {order_id}: need ${est_cost:,.2f}, have ${self.cash:,.2f}"
            )

        # Process by order type
        if order.order_type == OrderType.MARKET:
            fill_price = self.cost_model.calculate_fill_price(ref_price, is_buy=is_buy)
            comm = self.cost_model.calculate_commission(order.quantity, fill_price, is_buy=is_buy)

            res = OrderResult(
                broker_order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                status=OrderStatus.FILLED,
                quantity=order.quantity,
                filled_qty=order.quantity,
                filled_avg_price=fill_price,
                submitted_at=now,
                filled_at=now,
                raw_response={"commission": comm, "fill_price": fill_price, "client_order_id": order.client_order_id},
            )
            self._orders[order_id] = res
            self._apply_fill(order, order_id, fill_price, order.quantity, comm)

            # Check if this was a bracket order
            if order.order_class == OrderClass.BRACKET:
                self._create_bracket_children(order, order_id, fill_price)

            return res

        else:
            # LIMIT or STOP: Enter SUBMITTED state
            res = OrderResult(
                broker_order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                status=OrderStatus.SUBMITTED,
                quantity=order.quantity,
                filled_qty=0.0,
                submitted_at=now,
                raw_response={"client_order_id": order.client_order_id},
            )
            self._orders[order_id] = res
            return res

    async def cancel_order(self, broker_order_id: str) -> OrderResult:
        order = self._orders.get(broker_order_id)
        if not order:
            raise OrderRejectedError(f"Order '{broker_order_id}' not found.", order_id=broker_order_id)

        if not OrderStateMachine.can_cancel(order.status):
            raise OrderRejectedError(
                f"Cannot cancel order {broker_order_id} in {order.status.value} state.",
                order_id=broker_order_id,
            )

        order.status = OrderStateMachine.transition(order.status, OrderStatus.CANCELLED, broker_order_id)
        logger.info("paper_order_cancelled", order_id=broker_order_id)

        # If this order had an OCO sibling, cancel it too
        sibling_id = self._oco_siblings.pop(broker_order_id, None)
        if sibling_id and sibling_id in self._orders:
            sib = self._orders[sibling_id]
            if OrderStateMachine.can_cancel(sib.status):
                sib.status = OrderStatus.CANCELLED
                logger.info("oco_sibling_cancelled", sibling_id=sibling_id)

        return order

    async def cancel_all_orders(self) -> list[OrderResult]:
        """Emergency method called by Kill Switch to cancel all pending orders."""
        cancelled = []
        for order_id, order in list(self._orders.items()):
            if OrderStateMachine.can_cancel(order.status):
                order.status = OrderStatus.CANCELLED
                cancelled.append(order)
        logger.warning("all_paper_orders_cancelled", count=len(cancelled))
        return cancelled

    async def get_order(self, broker_order_id: str) -> OrderResult:
        order = self._orders.get(broker_order_id)
        if not order:
            raise OrderRejectedError(f"Order '{broker_order_id}' not found", order_id=broker_order_id)
        return order

    async def get_order_by_client_order_id(self, client_order_id: str) -> OrderResult | None:
        """Find order by client_order_id for idempotency verification and crash recovery."""
        for order in self._orders.values():
            if order.raw_response and order.raw_response.get("client_order_id") == client_order_id:
                return order
        return None

    async def get_open_orders(self) -> list[OrderResult]:
        return [
            o for o in self._orders.values()
            if OrderStateMachine.is_active(o.status)
        ]

    # ── Positions ────────────────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    # ── Execution Helpers ────────────────────────────────────────────

    def _apply_fill(
        self,
        order: OrderRequest,
        order_id: str,
        fill_price: float,
        quantity: float,
        commission: float,
    ) -> None:
        sym = order.symbol
        is_buy = order.side == OrderSide.BUY
        trade_val = quantity * fill_price

        if is_buy:
            self.cash -= (trade_val + commission)
            if sym in self._positions:
                existing = self._positions[sym]
                if existing.side == "long":
                    # Add to long
                    total_qty = existing.quantity + quantity
                    total_cost = (existing.quantity * existing.avg_entry_price) + trade_val
                    existing.avg_entry_price = round(total_cost / total_qty, 2)
                    existing.quantity = total_qty
                else:
                    # Closing short
                    close_qty = min(quantity, existing.quantity)
                    cost_basis = close_qty * existing.avg_entry_price
                    exit_val = close_qty * fill_price
                    net_pnl = cost_basis - exit_val - commission
                    pnl_pct = ((existing.avg_entry_price - fill_price) / existing.avg_entry_price * 100.0) if existing.avg_entry_price > 0 else 0.0
                    self.realized_pnl += net_pnl
                    reason = "TAKE_PROFIT" if order.order_type == OrderType.LIMIT else ("STOP_LOSS" if order.order_type == OrderType.STOP else "MARKET_CLOSE")
                    self._trade_history.append({
                        "trade_id": f"tr_{uuid4().hex[:8]}",
                        "order_id": order_id[:12],
                        "symbol": sym,
                        "side": "SHORT",
                        "qty": close_qty,
                        "entry_price": round(existing.avg_entry_price, 2),
                        "exit_price": round(fill_price, 2),
                        "notional": round(exit_val, 2),
                        "commission": round(commission, 2),
                        "pnl": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "outcome": "WIN" if net_pnl >= 0 else "LOSS",
                        "reason": reason,
                        "closed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    logger.info(
                        "trade_closed_realized",
                        symbol=sym,
                        side="SHORT",
                        pnl=round(net_pnl, 2),
                        outcome="WIN" if net_pnl >= 0 else "LOSS",
                        reason=reason,
                    )
                    if quantity >= existing.quantity:
                        del self._positions[sym]
                    else:
                        existing.quantity -= quantity
                        existing.market_value = round(existing.quantity * fill_price, 2)
            else:
                self._positions[sym] = Position(
                    symbol=sym,
                    side="long",
                    quantity=quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    market_value=round(trade_val, 2),
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                )
        else:
            # Sell
            self.cash += (trade_val - commission)
            if sym in self._positions:
                existing = self._positions[sym]
                if existing.side == "long":
                    # Closing long
                    close_qty = min(quantity, existing.quantity)
                    cost_basis = close_qty * existing.avg_entry_price
                    exit_val = close_qty * fill_price
                    net_pnl = exit_val - cost_basis - commission
                    pnl_pct = ((fill_price - existing.avg_entry_price) / existing.avg_entry_price * 100.0) if existing.avg_entry_price > 0 else 0.0
                    self.realized_pnl += net_pnl
                    reason = "TAKE_PROFIT" if order.order_type == OrderType.LIMIT else ("STOP_LOSS" if order.order_type == OrderType.STOP else "MARKET_CLOSE")
                    self._trade_history.append({
                        "trade_id": f"tr_{uuid4().hex[:8]}",
                        "order_id": order_id[:12],
                        "symbol": sym,
                        "side": "LONG",
                        "qty": close_qty,
                        "entry_price": round(existing.avg_entry_price, 2),
                        "exit_price": round(fill_price, 2),
                        "notional": round(exit_val, 2),
                        "commission": round(commission, 2),
                        "pnl": round(net_pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "outcome": "WIN" if net_pnl >= 0 else "LOSS",
                        "reason": reason,
                        "closed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    logger.info(
                        "trade_closed_realized",
                        symbol=sym,
                        side="LONG",
                        pnl=round(net_pnl, 2),
                        outcome="WIN" if net_pnl >= 0 else "LOSS",
                        reason=reason,
                    )
                    if quantity >= existing.quantity:
                        del self._positions[sym]
                    else:
                        existing.quantity -= quantity
                        existing.market_value = round(existing.quantity * fill_price, 2)
                else:
                    # Add to short
                    total_qty = existing.quantity + quantity
                    total_cost = (existing.quantity * existing.avg_entry_price) + trade_val
                    existing.avg_entry_price = round(total_cost / total_qty, 2)
                    existing.quantity = total_qty
            else:
                # Open short
                self._positions[sym] = Position(
                    symbol=sym,
                    side="short",
                    quantity=quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    market_value=round(trade_val, 2),
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                )
        self._save_persisted_state()

    def _create_bracket_children(self, parent_order: OrderRequest, parent_id: str, entry_fill: float) -> None:
        """Create child Take-Profit and Stop-Loss orders for a filled Bracket order."""
        child_ids = []
        opposite_side = OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY

        tp_id = f"paper_tp_{uuid4().hex[:10]}"
        sl_id = f"paper_sl_{uuid4().hex[:10]}"

        if parent_order.take_profit_price:
            tp_order = OrderRequest(
                symbol=parent_order.symbol,
                side=opposite_side,
                order_type=OrderType.LIMIT,
                quantity=parent_order.quantity,
                limit_price=parent_order.take_profit_price,
                order_class=OrderClass.OCO,
            )
            self._order_requests[tp_id] = tp_order
            self._orders[tp_id] = OrderResult(
                broker_order_id=tp_id,
                symbol=parent_order.symbol,
                side=opposite_side,
                order_type=OrderType.LIMIT,
                status=OrderStatus.SUBMITTED,
                quantity=parent_order.quantity,
            )
            child_ids.append(tp_id)

        if parent_order.stop_loss_price:
            sl_order = OrderRequest(
                symbol=parent_order.symbol,
                side=opposite_side,
                order_type=OrderType.STOP,
                quantity=parent_order.quantity,
                stop_price=parent_order.stop_loss_price,
                order_class=OrderClass.OCO,
            )
            self._order_requests[sl_id] = sl_order
            self._orders[sl_id] = OrderResult(
                broker_order_id=sl_id,
                symbol=parent_order.symbol,
                side=opposite_side,
                order_type=OrderType.STOP,
                status=OrderStatus.SUBMITTED,
                quantity=parent_order.quantity,
            )
            child_ids.append(sl_id)

        # Link OCO siblings if both exist
        if len(child_ids) == 2:
            self._oco_siblings[tp_id] = sl_id
            self._oco_siblings[sl_id] = tp_id

        self._bracket_children[parent_id] = child_ids
        logger.info("bracket_orders_armed", parent_id=parent_id, children=child_ids)

    def _match_pending_orders(self, symbol: str, close: float, high: float, low: float) -> None:
        """Evaluate pending LIMIT and STOP orders against recent price bar."""
        now = datetime.now(timezone.utc)

        for order_id, order in list(self._orders.items()):
            if order.symbol != symbol or not OrderStateMachine.is_active(order.status):
                continue

            req = self._order_requests.get(order_id)
            if not req:
                continue

            filled = False
            fill_price = close
            is_buy = req.side == OrderSide.BUY

            # Match Buy Limit
            if req.order_type == OrderType.LIMIT and is_buy and req.limit_price and low <= req.limit_price:
                fill_price = req.limit_price
                filled = True

            # Match Sell Limit (e.g. Take Profit)
            elif req.order_type == OrderType.LIMIT and not is_buy and req.limit_price and high >= req.limit_price:
                fill_price = req.limit_price
                filled = True

            # Match Buy Stop
            elif req.order_type == OrderType.STOP and is_buy and req.stop_price and high >= req.stop_price:
                fill_price = req.stop_price
                filled = True

            # Match Sell Stop (e.g. Stop Loss)
            elif req.order_type == OrderType.STOP and not is_buy and req.stop_price and low <= req.stop_price:
                fill_price = req.stop_price
                filled = True

            if filled:
                actual_fill = self.cost_model.calculate_fill_price(fill_price, is_buy=is_buy)
                comm = self.cost_model.calculate_commission(req.quantity, actual_fill, is_buy=is_buy)

                order.status = OrderStatus.FILLED
                order.filled_qty = req.quantity
                order.filled_avg_price = actual_fill
                order.filled_at = now

                self._apply_fill(req, order_id, actual_fill, req.quantity, comm)
                logger.info("paper_order_filled", order_id=order_id, fill_price=actual_fill)

                # OCO Cancellation: If this was part of an OCO pair, cancel the sibling immediately
                sibling_id = self._oco_siblings.pop(order_id, None)
                if sibling_id and sibling_id in self._orders:
                    sib = self._orders[sibling_id]
                    if OrderStateMachine.can_cancel(sib.status):
                        sib.status = OrderStatus.CANCELLED
                        logger.info("oco_sibling_auto_cancelled", sibling_id=sibling_id)

    # ── Reconciliation ───────────────────────────────────────────────

    async def reconcile(self) -> dict[str, Any]:
        """Verify internal consistency between cash, positions, and orders based on Broker Truth."""
        account = await self.get_account()
        discrepancies = []

        for sym, pos in list(self._positions.items()):
            if pos.quantity <= 0:
                discrepancies.append(f"Position for {sym} has non-positive quantity: {pos.quantity}")
                del self._positions[sym]

        pending_reconciliations = []

        # Broker Truth: Reconcile positions against filled orders
        for o_id, order in self._orders.items():
            if order.status == OrderStatus.FILLED:
                sym = order.symbol
                qty = order.filled_qty or order.quantity
                price = order.filled_avg_price

                # Audit Rule: NEVER invent a synthetic price ($100.0 or req_limit).
                # If filled_avg_price is missing, halt position creation and mark RECONCILIATION_PENDING.
                if price is None or price <= 0.0:
                    error_msg = (
                        f"RECONCILIATION_PENDING: Filled order {o_id} for {sym} has unknown execution price "
                        f"(filled_avg_price={order.filled_avg_price}). Position reconciliation halted to prevent P&L corruption."
                    )
                    logger.error("reconcile_missing_execution_price", order_id=o_id, symbol=sym)
                    pending_reconciliations.append({
                        "order_id": o_id,
                        "symbol": sym,
                        "status": "RECONCILIATION_PENDING",
                        "reason": "UNKNOWN_EXECUTION_PRICE",
                    })
                    discrepancies.append(error_msg)
                    continue

                if order.side == OrderSide.BUY:
                    if sym not in self._positions:
                        discrepancies.append(f"Filled BUY order {o_id} for {sym} had no local position; reconciled at broker price ${price:,.2f}.")
                        self._positions[sym] = Position(
                            symbol=sym,
                            quantity=qty,
                            avg_entry_price=price,
                            current_price=price,
                            market_value=qty * price,
                            cost_basis=qty * price,
                            unrealized_pnl=0.0,
                            unrealized_pnl_pct=0.0,
                            side="long",
                        )
                        self._reconciled_order_ids.add(o_id)
                    elif o_id not in self._reconciled_order_ids:
                        pos = self._positions[sym]
                        new_qty = pos.quantity + qty
                        new_cost = (pos.quantity * pos.avg_entry_price) + (qty * price)
                        pos.avg_entry_price = round(new_cost / new_qty, 4)
                        pos.quantity = new_qty
                        pos.market_value = new_qty * price
                        discrepancies.append(f"Filled BUY order {o_id} for {sym} increased position to {new_qty} shares (new avg entry: ${pos.avg_entry_price:,.4f}).")
                        self._reconciled_order_ids.add(o_id)
                elif order.side == OrderSide.SELL:
                    if sym in self._positions and o_id not in self._reconciled_order_ids:
                        pos = self._positions[sym]
                        if pos.side == "long":
                            # Weighted average cost basis for realized P&L
                            realized = qty * (price - pos.avg_entry_price)
                            self.realized_pnl += realized
                            self.cash += (qty * price)
                            if qty >= pos.quantity:
                                discrepancies.append(f"Filled SELL order {o_id} for {sym} closes remaining position of {pos.quantity} shares.")
                                del self._positions[sym]
                            else:
                                discrepancies.append(f"Filled SELL order {o_id} for {sym} reduces position from {pos.quantity} by {qty} shares.")
                                pos.quantity -= qty
                                pos.market_value = pos.quantity * price
                        self._reconciled_order_ids.add(o_id)

        status = "clean"
        if pending_reconciliations:
            status = "RECONCILIATION_PENDING"
        elif discrepancies:
            status = "reconciled_with_broker"

        return {
            "status": status,
            "account_equity": account.equity,
            "cash": account.cash,
            "open_positions": len(self._positions),
            "total_orders_tracked": len(self._orders),
            "discrepancies": discrepancies,
            "pending_reconciliations": pending_reconciliations,
            "provider": "paper_broker",
        }

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "healthy" if self._connected else "disconnected",
            "provider": "paper_broker",
            "open_orders": len(await self.get_open_orders()),
            "positions": len(self._positions),
        }

    # ── History & Reset ───────────────────────────────────────────────

    def get_trade_history(self) -> list[dict[str, Any]]:
        """Return all closed trades in reverse chronological order (newest first)."""
        return list(reversed(self._trade_history))

    def reset_account(self, initial_cash: float | None = None) -> None:
        """Resets the paper account back to clean baseline state ($100,000.00)."""
        if initial_cash is not None:
            self.initial_cash = initial_cash
        self.cash = self.initial_cash
        self._positions.clear()
        self._orders.clear()
        self._order_requests.clear()
        self._bracket_children.clear()
        self._oco_siblings.clear()
        self.realized_pnl = 0.0
        self._trade_history.clear()
        self._save_persisted_state()
        logger.info("paper_broker_account_reset", cash=self.cash)
