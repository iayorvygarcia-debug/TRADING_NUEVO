"""
Unit Tests for Broker Engine, Order State Machine, Paper Broker, and Execution Engine.

Tests:
1. OrderStateMachine (valid/invalid transitions, terminal states)
2. PaperBroker (Market, Limit, Stop, Bracket OCO orders, price matching, cancel all)
3. ExecutionEngine (Risk Engine routing, Bracket order generation)
"""

from __future__ import annotations

from unittest.mock import patch
import pytest

from services.broker_engine.base import (
    OrderClass,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from services.broker_engine.order_state_machine import (
    OrderStateMachine,
    InvalidOrderStateTransitionError,
)
from services.broker_engine.paper_broker import PaperBroker
from services.broker_engine.execution_engine import ExecutionEngine
from services.risk_engine.engine import RiskEngine
from services.strategy_engine.signal_combiner import CombinedSignal
from strategies.base import SignalDirection


# ── 1. Order State Machine Tests ─────────────────────────────────────

class TestOrderStateMachine:

    def test_valid_transitions(self) -> None:
        # CREATED -> APPROVED -> SUBMITTED -> FILLED
        s1 = OrderStateMachine.transition(OrderStatus.CREATED, OrderStatus.APPROVED)
        assert s1 == OrderStatus.APPROVED

        s2 = OrderStateMachine.transition(s1, OrderStatus.SUBMITTED)
        assert s2 == OrderStatus.SUBMITTED

        s3 = OrderStateMachine.transition(s2, OrderStatus.FILLED)
        assert s3 == OrderStatus.FILLED

    def test_invalid_transition_raises(self) -> None:
        # CREATED directly to FILLED is illegal
        with pytest.raises(InvalidOrderStateTransitionError):
            OrderStateMachine.transition(OrderStatus.CREATED, OrderStatus.FILLED)

    def test_terminal_state_immutable(self) -> None:
        # Once FILLED, cannot be CANCELLED or REJECTED
        with pytest.raises(InvalidOrderStateTransitionError):
            OrderStateMachine.transition(OrderStatus.FILLED, OrderStatus.CANCELLED)

        with pytest.raises(InvalidOrderStateTransitionError):
            OrderStateMachine.transition(OrderStatus.CANCELLED, OrderStatus.FILLED)

    def test_predicates(self) -> None:
        assert OrderStateMachine.is_terminal(OrderStatus.FILLED) is True
        assert OrderStateMachine.is_terminal(OrderStatus.CANCELLED) is True
        assert OrderStateMachine.is_terminal(OrderStatus.SUBMITTED) is False

        assert OrderStateMachine.is_active(OrderStatus.SUBMITTED) is True
        assert OrderStateMachine.is_active(OrderStatus.FILLED) is False

        assert OrderStateMachine.can_cancel(OrderStatus.SUBMITTED) is True
        assert OrderStateMachine.can_cancel(OrderStatus.FILLED) is False


# ── 2. Paper Broker Tests ────────────────────────────────────────────

class TestPaperBroker:

    @pytest.fixture
    def broker(self) -> PaperBroker:
        return PaperBroker(initial_cash=50000.0)

    @pytest.mark.anyio
    async def test_connect_and_account(self, broker: PaperBroker) -> None:
        await broker.connect()
        acc = await broker.get_account()
        assert acc.cash == 50000.0
        assert acc.equity == 50000.0
        assert acc.trading_mode == "paper"
        await broker.disconnect()

    @pytest.mark.anyio
    async def test_market_order_fill(self, broker: PaperBroker) -> None:
        await broker.connect()
        broker.update_market_price("AAPL", 150.0)

        # Submit 10 shares MARKET order
        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
        )
        res = await broker.submit_order(req)
        assert res.status == OrderStatus.FILLED
        assert res.filled_qty == 10.0
        assert res.filled_avg_price is not None and res.filled_avg_price > 150.0  # Slippage/spread

        # Verify position created
        pos = await broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 10.0
        assert pos.side == "long"

        # Verify cash deducted
        acc = await broker.get_account()
        assert acc.cash < 50000.0

        await broker.disconnect()

    @pytest.mark.anyio
    async def test_bracket_order_and_oco_resolution(self, broker: PaperBroker) -> None:
        """Test bracket order arms TP/SL and filling TP cancels SL."""
        await broker.connect()
        broker.update_market_price("TSLA", 200.0)

        # Place Bracket Order: Entry @ 200, TP @ 220, SL @ 190
        req = OrderRequest(
            symbol="TSLA",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            order_class=OrderClass.BRACKET,
            take_profit_price=220.0,
            stop_loss_price=190.0,
        )
        parent_res = await broker.submit_order(req)
        assert parent_res.status == OrderStatus.FILLED

        # Check that 2 open orders exist (Take-Profit & Stop-Loss)
        open_orders = await broker.get_open_orders()
        assert len(open_orders) == 2
        tp_orders = [o for o in open_orders if o.order_type == OrderType.LIMIT]
        sl_orders = [o for o in open_orders if o.order_type == OrderType.STOP]
        assert len(tp_orders) == 1
        assert len(sl_orders) == 1

        # Simulate price moving to 225 (crossing Take-Profit @ 220)
        broker.update_market_price("TSLA", 225.0, high=225.0, low=198.0)

        # Take Profit order should be FILLED
        tp_res = await broker.get_order(tp_orders[0].broker_order_id)
        assert tp_res.status == OrderStatus.FILLED

        # Sibling Stop Loss order should be CANCELLED automatically (OCO)
        sl_res = await broker.get_order(sl_orders[0].broker_order_id)
        assert sl_res.status == OrderStatus.CANCELLED

        # No open orders should remain
        assert len(await broker.get_open_orders()) == 0

        # Position should now be closed
        pos = await broker.get_position("TSLA")
        assert pos is None

        # Cash should be higher than initial $50,000 (profit realized!)
        acc = await broker.get_account()
        assert acc.cash > 50000.0

        await broker.disconnect()

    @pytest.mark.anyio
    async def test_cancel_all_orders(self, broker: PaperBroker) -> None:
        await broker.connect()
        broker.update_market_price("NVDA", 100.0)

        # Place 2 pending limit orders
        req1 = OrderRequest(symbol="NVDA", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=5, limit_price=90.0)
        req2 = OrderRequest(symbol="NVDA", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=5, limit_price=85.0)

        await broker.submit_order(req1)
        await broker.submit_order(req2)
        assert len(await broker.get_open_orders()) == 2

        # Emergency cancel all
        cancelled = await broker.cancel_all_orders()
        assert len(cancelled) == 2
        assert len(await broker.get_open_orders()) == 0

        await broker.disconnect()


# ── 3. Execution Engine Tests ────────────────────────────────────────

class TestExecutionEngine:

    @pytest.mark.anyio
    async def test_execute_approved_signal(self) -> None:
        broker = PaperBroker(initial_cash=100000.0)
        await broker.connect()
        broker.update_market_price("MSFT", 400.0)

        engine = ExecutionEngine(broker=broker, risk_engine=RiskEngine())

        signal = CombinedSignal(
            symbol="MSFT",
            direction=SignalDirection.BUY,
            composite_score=0.85,
            confidence=0.80,
            consensus_ratio=1.0,
            contributing_strategies=["trend_following"],
            suggested_stop=380.0,
            suggested_target=440.0,
            is_approved=True,
        )

        with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
            res = await engine.execute_signal(
                signal=signal,
                current_price=400.0,
                default_shares=10.0,
            )
            assert res is not None
            assert res.status == OrderStatus.FILLED
            assert res.symbol == "MSFT"

        # Position should exist with bracket legs armed
        pos = await broker.get_position("MSFT")
        assert pos is not None
        assert pos.quantity == 10.0

        await broker.disconnect()

    @pytest.mark.anyio
    async def test_skip_non_actionable_signal(self) -> None:
        broker = PaperBroker()
        await broker.connect()
        engine = ExecutionEngine(broker=broker)

        # Unapproved signal
        signal = CombinedSignal(
            symbol="MSFT",
            direction=SignalDirection.HOLD,
            composite_score=0.40,
            confidence=0.40,
            consensus_ratio=0.0,
            contributing_strategies=[],
            is_approved=False,
        )

        res = await engine.execute_signal(signal=signal, current_price=400.0)
        assert res is None  # Skipped cleanly
        await broker.disconnect()
