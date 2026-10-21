"""
AI Quant Trading Platform — Order Execution Engine.

The execution gateway that routes actionable strategy signals into the broker.
Crucial invariant:
    Signal -> Risk Engine Gatekeeper -> OrderRequest -> BrokerInterface
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from core.config import get_settings
from core.exceptions import BrokerStateUnknownError
from core.logging import get_logger
from services.broker_engine.base import (
    BrokerInterface,
    OrderAction,
    OrderClass,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from services.risk_engine.engine import RiskEngine, get_risk_engine
from services.risk_engine.models import PortfolioRiskState
from services.strategy_engine.signal_combiner import CombinedSignal
from strategies.base import SignalDirection

logger = get_logger("execution_engine")


class ExecutionEngine:
    """
    Coordinates pre-trade risk evaluation and order submission to the broker.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        import asyncio
        self.broker = broker
        self.risk_engine = risk_engine or get_risk_engine()
        self.settings = get_settings()
        self._symbol_locks: dict[str, asyncio.Lock] = {}

    def _get_symbol_lock(self, symbol: str):
        import asyncio
        if symbol not in self._symbol_locks:
            self._symbol_locks[symbol] = asyncio.Lock()
        return self._symbol_locks[symbol]

    def _generate_client_order_id(
        self,
        signal: CombinedSignal,
        order_side: OrderSide,
        action: OrderAction,
    ) -> str:
        """
        Generate a deterministic, unique client_order_id identifying the logical operation.
        Includes signal properties and strategy hash so distinct signals produce distinct IDs,
        while the exact same logical signal processed by multiple workers produces the exact same ID.
        """
        import hashlib
        strat_str = ",".join(sorted(getattr(signal, "contributing_strategies", [])))
        score_str = f"{getattr(signal, 'composite_score', 0.0):.3f}"
        ts = getattr(signal, "timestamp", None)
        ts_str = ts.isoformat() if ts else "notime"
        sig_fingerprint = hashlib.sha256(
            f"{signal.symbol}_{order_side.value}_{action.value}_{score_str}_{strat_str}_{ts_str}".encode()
        ).hexdigest()[:10]
        return f"agy_{signal.symbol}_{order_side.value[:1]}_{action.value[:3]}_{sig_fingerprint}"

    async def execute_signal(
        self,
        signal: CombinedSignal,
        current_price: float,
        sector: str = "technology",
        current_spread_pct: float = 0.001,
        order_type: OrderType = OrderType.MARKET,
        default_shares: float | None = None,
        extended_hours: bool = False,
        limit_price: float | None = None,
        natr_14: float | None = None,
        size_multiplier: float = 1.0,
        action: OrderAction = OrderAction.ENTRY,
        is_exit: bool = False,
    ) -> OrderResult | None:
        """
        Process a strategy signal through Risk Engine and submit to Broker if approved.
        Supports explicit action (ENTRY vs EXIT/REDUCE) and propagates size_multiplier.
        """
        if not signal.is_actionable:
            logger.info("signal_not_actionable_skipping", symbol=signal.symbol)
            return None

        is_exit_action = is_exit or (action in {OrderAction.EXIT, OrderAction.REDUCE})

        async with self._get_symbol_lock(signal.symbol):
            # 1. Fetch current account state and active open orders with fail-safe error handling
            try:
                account = await self.broker.get_account()
                positions = await self.broker.get_positions()
                open_orders = await self.broker.get_open_orders()
            except (BrokerStateUnknownError, Exception) as exc:
                logger.error(
                    "execution_engine_broker_state_unknown_halting",
                    symbol=signal.symbol,
                    error=str(exc),
                )
                return None

            existing_pos = next((p for p in positions if p.symbol == signal.symbol), None)

            # Idempotency & Action validation
            if not is_exit_action:
                # ENTRY: Reject if position already exists in this symbol to prevent accidental double-entry
                if existing_pos:
                    logger.warning(
                        "position_already_exists_for_symbol_skipping_duplicate",
                        symbol=signal.symbol,
                        side=signal.direction.name,
                    )
                    return None

                pending_symbols = {
                    o.symbol for o in open_orders
                    if getattr(o, "status", None) in {
                        OrderStatus.SUBMITTED,
                        OrderStatus.PARTIALLY_FILLED,
                        OrderStatus.APPROVED,
                        OrderStatus.CREATED,
                    }
                }
                if signal.symbol in pending_symbols:
                    logger.warning(
                        "order_already_pending_for_symbol_skipping_duplicate",
                        symbol=signal.symbol,
                        side=signal.direction.name,
                    )
                    return None
            else:
                # EXIT / REDUCE: Position must exist in order to be closed or reduced
                if not existing_pos:
                    logger.warning(
                        "cannot_exit_nonexistent_position",
                        symbol=signal.symbol,
                        action=action.value,
                    )
                    return None

            # Real capital baseline: respect the $1,000 operational capital
            settings = getattr(self, "settings", None) or get_settings()
            use_op_cap = getattr(settings.risk, "use_operational_capital", True)
            if use_op_cap:
                account_equity = float(getattr(settings.risk, "operational_capital", 1000.0))
                max_pos_size_pct = float(getattr(settings.risk, "max_operational_position_size", 0.20))
            else:
                account_equity = float(account.equity) if (account and getattr(account, "equity", 0) > 0) else float(settings.risk.initial_capital)
                max_pos_size_pct = float(getattr(settings.risk, "max_position_size", 0.05))

            risk_pct = float(getattr(settings.risk, "risk_per_trade", getattr(settings.risk, "risk_per_trade_pct", 0.01)))

            # Build session-aware PortfolioRiskState with true daily starting & peak equity
            if hasattr(self.risk_engine, "get_or_create_portfolio_state"):
                portfolio_state = self.risk_engine.get_or_create_portfolio_state(
                    equity=account.equity,
                    cash=account.cash,
                    open_positions_count=len(positions),
                )
            else:
                portfolio_state = PortfolioRiskState(
                    equity=account.equity,
                    peak_equity=account.equity,
                    cash=account.cash,
                    daily_starting_equity=account.equity,
                    open_positions_count=len(positions),
                )

            side_str = "buy" if signal.direction == SignalDirection.BUY else "sell"
            order_side = OrderSide.BUY if signal.direction == SignalDirection.BUY else OrderSide.SELL

            # Explicit Action Logic: EXIT and REDUCE manage existing position side and quantity
            if is_exit_action and existing_pos:
                # If closing/reducing a LONG, order is SELL; if closing/reducing a SHORT, order is BUY
                if existing_pos.side.lower() == "long":
                    side_str = "sell"
                    order_side = OrderSide.SELL
                else:
                    side_str = "buy"
                    order_side = OrderSide.BUY

                if action == OrderAction.EXIT:
                    order_shares = abs(existing_pos.quantity)
                else:  # REDUCE
                    explicit_qty = float(default_shares) if default_shares is not None else abs(existing_pos.quantity * 0.5)
                    order_shares = min(abs(existing_pos.quantity), max(1.0, explicit_qty))
            else:
                # ENTRY: Fail-safe check on size_multiplier. If <= 0, ABORT. NEVER force 1 share on zero multiplier.
                if size_multiplier <= 0.0:
                    logger.info("execution_aborted_zero_size_multiplier", symbol=signal.symbol, size_multiplier=size_multiplier)
                    return None

                if default_shares is not None:
                    base_shares = float(default_shares)
                elif current_price > 0:
                    from services.risk_engine.position_sizing import risk_based_shares
                    # On a $1,000 account, skip assets whose single share exceeds $450 (e.g. MELI, MSFT)
                    if use_op_cap and current_price > (account_equity * 0.45):
                        logger.info("execution_aborted_price_exceeds_operational_budget", symbol=signal.symbol, price=current_price, max_price=account_equity * 0.45)
                        return None

                    base_shares = risk_based_shares(
                        capital=account_equity,
                        entry_price=current_price,
                        stop_price=signal.suggested_stop,
                        risk_per_trade_pct=risk_pct,
                        max_notional_pct=max_pos_size_pct,
                    )
                    if base_shares <= 0 and current_price <= (account_equity * 0.45):
                        stop_px = signal.suggested_stop
                        one_share_risk = abs(current_price - stop_px) if stop_px else (current_price * 0.02)
                        if one_share_risk <= (account_equity * 0.03):
                            base_shares = 1.0
                else:
                    base_shares = 2.0

                order_shares = float(math.floor(base_shares * size_multiplier))
                # If calculated quantity rounds down to 0, FAIL-SAFE to 0 orders. NEVER invent 1 share.
                if order_shares <= 0.0:
                    logger.info(
                        "execution_aborted_zero_shares_calculated",
                        symbol=signal.symbol,
                        base_shares=base_shares,
                        size_multiplier=size_multiplier,
                    )
                    return None

            # 2. Risk Engine Evaluation
            risk_result = self.risk_engine.evaluate_order(
                symbol=signal.symbol,
                direction=side_str,
                quantity=order_shares,
                price=current_price,
                portfolio_state=portfolio_state,
                signal_score=signal.composite_score,
                current_spread_pct=current_spread_pct,
                sector=sector,
                is_exit=is_exit_action,
                extended_hours=extended_hours,
                natr_14=natr_14,
            )

            if not risk_result.approved:
                logger.warning(
                    "order_rejected_by_risk_engine",
                    symbol=signal.symbol,
                    violations=risk_result.violations,
                )
                return None

            # 3. Deterministic client_order_id identifying the logical operation
            deterministic_client_id = self._generate_client_order_id(signal, order_side, action)

            # Check if broker already has an order with this deterministic client_order_id (idempotency check)
            if hasattr(self.broker, "get_order_by_client_order_id"):
                try:
                    existing_order = await self.broker.get_order_by_client_order_id(deterministic_client_id)
                    if existing_order:
                        logger.warning(
                            "order_already_exists_for_client_order_id_adopting",
                            client_order_id=deterministic_client_id,
                            broker_order_id=existing_order.broker_order_id,
                        )
                        return existing_order
                except Exception:
                    pass

            # Build OrderRequest with Bracket parameters if stops/targets exist and this is an entry
            is_bracket = bool(signal.suggested_stop and signal.suggested_target and not extended_hours and not is_exit_action)
            order_class = OrderClass.BRACKET if is_bracket else OrderClass.SIMPLE

            order_req = OrderRequest(
                symbol=signal.symbol,
                side=order_side,
                order_type=order_type,
                action=action,
                client_order_id=deterministic_client_id,
                quantity=risk_result.approved_quantity,
                limit_price=limit_price,
                order_class=order_class,
                extended_hours=extended_hours,
                take_profit_price=signal.suggested_target if is_bracket else None,
                stop_loss_price=signal.suggested_stop if is_bracket else None,
                strategy=",".join(signal.contributing_strategies),
            )

            # 4. Submit to Broker with lost-response recovery
            logger.info(
                "submitting_order_to_broker",
                symbol=signal.symbol,
                side=order_side.value,
                action=action.value,
                quantity=order_req.quantity,
                is_bracket=is_bracket,
                client_order_id=deterministic_client_id,
            )

            try:
                result = await self.broker.submit_order(order_req)
                return result
            except Exception as exc:
                # Network timeout/drop recovery: Verify if the broker actually accepted the order
                logger.warning(
                    "order_submission_failed_checking_client_order_id_recovery",
                    client_order_id=deterministic_client_id,
                    error=str(exc),
                )
                if hasattr(self.broker, "get_order_by_client_order_id"):
                    try:
                        recovered = await self.broker.get_order_by_client_order_id(deterministic_client_id)
                        if recovered:
                            logger.info(
                                "order_submission_recovered_from_network_timeout",
                                client_order_id=deterministic_client_id,
                                broker_order_id=recovered.broker_order_id,
                            )
                            return recovered
                    except Exception:
                        pass
                return None
