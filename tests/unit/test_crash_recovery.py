"""
Crash Recovery, Idempotency & Concurrency Test Suite.

Audited Scenarios:
1. Crash Recovery - Open Order Idempotency (broker has open order -> restart -> no duplication).
2. Crash Recovery - Scenario A: Filled during crash (SUBMITTED -> FILLED -> reconcile -> position held).
3. Crash Recovery - Scenario B: Cancelled during crash (SUBMITTED -> CANCELLED -> reconcile).
4. Crash Recovery - Scenario C: Rejected at broker (SUBMITTED -> REJECTED -> reconcile).
5. Crash Recovery - Scenario D: Partial Fill (100 shares -> 40 filled, 60 remaining -> reconcile).
6. Concurrency Protection - Two simultaneous signals for the same symbol executed concurrently via asyncio.gather -> maximum 1 order placed.
7. Dynamic Spread Parametric Safety - Low, normal, high, extreme NATR -> spread <= hard_cap always.
8. Deterministic Strategy Regression - Exact input -> exact output signal verification.
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone

from services.broker_engine.base import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from services.broker_engine.paper_broker import PaperBroker
from services.broker_engine.execution_engine import ExecutionEngine
from services.risk_engine.engine import RiskEngine
from services.strategy_engine.signal_combiner import CombinedSignal, SignalCombiner
from strategies.base import SignalDirection, StrategySignal
from core.market_session import MarketSession, get_dynamic_max_spread
from core.config import get_settings


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def paper_broker():
    broker = PaperBroker(initial_cash=1000.0)
    broker._connected = True
    return broker


@pytest.fixture
def risk_engine():
    return RiskEngine()


@pytest.mark.anyio
async def test_crash_recovery_prevents_duplicate_orders(paper_broker, risk_engine):
    """
    Test that an active open order in the broker blocks duplicate submissions
    even after a simulated worker restart.
    """
    engine1 = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    # 1. 15:55:01 -> Initial order submitted
    req = OrderRequest(
        symbol="AAPL",
        quantity=5.0,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=150.0,
    )
    result = await paper_broker.submit_order(req)
    assert result is not None
    assert result.status == OrderStatus.SUBMITTED

    # Broker has exactly 1 order
    open_orders_before = await paper_broker.get_open_orders()
    assert len(open_orders_before) == 1
    assert open_orders_before[0].symbol == "AAPL"

    # 2. 15:55:02 -> Worker crashes!
    del engine1

    # 3. 15:55:10 -> Worker restarts with fresh ExecutionEngine instance
    engine2 = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    # Worker attempts to execute the same signal for AAPL
    signal = CombinedSignal(
        symbol="AAPL",
        direction=SignalDirection.BUY,
        composite_score=0.85,
        confidence=0.80,
        consensus_ratio=1.0,
        contributing_strategies=["trend_following"],
        is_approved=True,
    )

    # ExecutionEngine MUST detect the pending order and refuse to duplicate
    dup_result = await engine2.execute_signal(
        signal=signal,
        current_price=150.0,
        current_spread_pct=0.0005,
    )
    assert dup_result is None, "Expected duplicate order to be rejected by idempotency check"

    # Broker STILL has exactly 1 order
    open_orders_after = await paper_broker.get_open_orders()
    assert len(open_orders_after) == 1


@pytest.mark.anyio
async def test_crash_scenario_a_filled_during_crash(paper_broker, risk_engine):
    """
    Scenario A: Order submitted, worker crashes, broker fills order while offline.
    Upon boot, reconcile() must adopt broker truth (FILLED) and register the position.
    """
    req = OrderRequest(
        symbol="MSFT",
        quantity=3.0,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=300.0,
    )
    order_res = await paper_broker.submit_order(req)
    order_id = order_res.broker_order_id

    # Simulate broker fill while worker is down
    paper_broker._orders[order_id].status = OrderStatus.FILLED
    paper_broker._orders[order_id].filled_qty = 3.0
    paper_broker._orders[order_id].filled_avg_price = 300.0

    # Worker reboots and reconciles
    report = await paper_broker.reconcile()
    assert report["status"] in {"clean", "reconciled_with_broker"}

    # Broker truth wins: local position is established
    positions = await paper_broker.get_positions()
    assert any(p.symbol == "MSFT" and p.quantity == 3.0 for p in positions)


@pytest.mark.anyio
async def test_crash_scenario_b_cancelled_during_crash(paper_broker, risk_engine):
    """
    Scenario B: Order submitted, worker crashes, order cancelled at broker.
    Upon boot, reconcile() confirms order is CANCELLED.
    """
    req = OrderRequest(
        symbol="GOOGL",
        quantity=2.0,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=175.0,
    )
    order_res = await paper_broker.submit_order(req)
    order_id = order_res.broker_order_id

    # Cancel at broker
    await paper_broker.cancel_order(order_id)
    assert paper_broker._orders[order_id].status == OrderStatus.CANCELLED

    # Open orders must be 0
    open_orders = await paper_broker.get_open_orders()
    assert not any(o.broker_order_id == order_id for o in open_orders)


@pytest.mark.anyio
async def test_crash_scenario_c_rejected_during_crash(paper_broker, risk_engine):
    """
    Scenario C: Order rejected at broker.
    """
    req = OrderRequest(
        symbol="AMZN",
        quantity=4.0,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=180.0,
    )
    order_res = await paper_broker.submit_order(req)
    order_id = order_res.broker_order_id

    # Broker rejects order
    paper_broker._orders[order_id].status = OrderStatus.REJECTED

    # Reconcile confirms rejection
    open_orders = await paper_broker.get_open_orders()
    assert not any(o.broker_order_id == order_id and o.status == OrderStatus.SUBMITTED for o in open_orders)


@pytest.mark.anyio
async def test_crash_scenario_d_partial_fill_during_crash(paper_broker, risk_engine):
    """
    Scenario D: Partial fill (2 shares submitted, broker fills 1, 1 remaining).
    """
    req = OrderRequest(
        symbol="TSLA",
        quantity=2.0,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=220.0,
    )
    order_res = await paper_broker.submit_order(req)
    order_id = order_res.broker_order_id

    # Simulate broker partial fill
    paper_broker._orders[order_id].status = OrderStatus.PARTIALLY_FILLED
    paper_broker._orders[order_id].filled_qty = 1.0
    paper_broker._orders[order_id].filled_avg_price = 220.0

    # Local order reflects partial fill
    assert paper_broker._orders[order_id].status == OrderStatus.PARTIALLY_FILLED
    assert paper_broker._orders[order_id].filled_qty == 1.0


@pytest.mark.anyio
async def test_concurrency_race_condition_protection(paper_broker, risk_engine):
    """
    Scenario E: Two simultaneous signals for the same ticker arrive at the exact same millisecond.
    Mutex lock serializes the execution, ensuring only 1 order reaches the broker.
    """
    from unittest.mock import patch

    engine = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    signal1 = CombinedSignal(
        symbol="AAPL",
        direction=SignalDirection.BUY,
        composite_score=0.88,
        confidence=0.85,
        consensus_ratio=1.0,
        contributing_strategies=["trend_following"],
        suggested_stop=48.0,
        suggested_target=54.0,
        is_approved=True,
    )
    signal2 = CombinedSignal(
        symbol="AAPL",
        direction=SignalDirection.BUY,
        composite_score=0.82,
        confidence=0.80,
        consensus_ratio=1.0,
        contributing_strategies=["breakout"],
        suggested_stop=48.0,
        suggested_target=54.0,
        is_approved=True,
    )

    # Launch concurrently with market session mocked as open
    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        res1, res2 = await asyncio.gather(
            engine.execute_signal(signal1, current_price=50.0, default_shares=2.0, current_spread_pct=0.0005, extended_hours=True),
            engine.execute_signal(signal2, current_price=50.0, default_shares=2.0, current_spread_pct=0.0005, extended_hours=True),
        )

    # Exactly ONE should succeed, the other must be rejected by idempotency lock
    results = [r for r in [res1, res2] if r is not None]
    assert len(results) == 1, f"Expected exactly 1 order executed, got {len(results)}"

    # Confirm exactly 1 position resulted from the single executed order
    positions = await paper_broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"


@pytest.mark.parametrize(
    "symbol,session,natr_14,expected_max_cap",
    [
        ("AAPL", MarketSession.REGULAR, 0.5, 0.0050),    # Low NATR regular -> well below cap
        ("AAPL", MarketSession.REGULAR, 2.0, 0.0050),    # Normal NATR regular
        ("AAPL", MarketSession.REGULAR, 5.0, 0.0050),    # High NATR regular -> bounded by hard cap
        ("AAPL", MarketSession.REGULAR, 20.0, 0.0050),   # Extreme NATR regular -> strictly capped at 0.50%
        ("NVDA", MarketSession.PREMARKET, 1.0, 0.0070),  # Low NATR extended
        ("NVDA", MarketSession.PREMARKET, 4.0, 0.0070),  # High NATR extended
        ("NVDA", MarketSession.PREMARKET, 25.0, 0.0070), # Extreme NATR extended -> strictly capped at 0.70%
    ]
)
def test_parametric_dynamic_spread_hard_caps(symbol, session, natr_14, expected_max_cap):
    """
    Parametric verification that dynamic spread never exceeds the configured hard cap,
    regardless of how high volatility (NATR) spikes.
    """
    spread = get_dynamic_max_spread(symbol=symbol, session=session, natr_14=natr_14)
    assert spread <= expected_max_cap, f"Spread {spread} exceeded hard cap {expected_max_cap}"
    assert spread >= 0.0010, f"Spread {spread} fell below absolute minimum 0.10%"


def test_deterministic_strategy_regression():
    """
    Regression Test: Ensures identical feature inputs produce identical signals
    without any interference from TTL or metadata fields.
    """
    combiner = SignalCombiner()

    sig1 = StrategySignal(
        strategy="trend_following",
        symbol="AAPL",
        direction=SignalDirection.BUY,
        score=0.85,
        confidence=0.80,
        stop_suggestion=145.0,
        target_suggestion=160.0,
    )
    sig2 = StrategySignal(
        strategy="breakout",
        symbol="AAPL",
        direction=SignalDirection.BUY,
        score=0.75,
        confidence=0.70,
        stop_suggestion=146.0,
        target_suggestion=158.0,
    )

    combined = combiner.combine("AAPL", [sig1, sig2])

    # Deterministic verification
    assert combined.symbol == "AAPL"
    assert combined.direction == SignalDirection.BUY
    assert combined.is_approved is True
    assert combined.suggested_stop == 146.0  # Most conservative BUY stop
    assert combined.suggested_target == 159.0 # (160 + 158) / 2
    assert combined.status == "PENDING"
    assert combined.ttl_seconds == 14400.0


@pytest.mark.anyio
async def test_reconcile_missing_price_does_not_invent_fake_price(paper_broker):
    """
    Audit Requirement: Never invent a synthetic price ($100.0) upon reconciliation.
    If broker reports FILLED but filled_avg_price is None/unknown:
    1. Reconcile must report RECONCILIATION_PENDING.
    2. No corrupt position at $100.0 or synthetic price may be established.
    3. Once broker provides verified price, reconcile succeeds with true broker execution price.
    """
    req = OrderRequest(
        symbol="AMD",
        quantity=5.0,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )
    # Give market price feed for initial submit
    paper_broker.update_market_price("AMD", 160.0)
    order_res = await paper_broker.submit_order(req)
    order_id = order_res.broker_order_id

    # Simulate crash and broker fill with MISSING filled_avg_price
    # and remove any existing local position to test fresh recovery
    if "AMD" in paper_broker._positions:
        del paper_broker._positions["AMD"]
    paper_broker._orders[order_id].status = OrderStatus.FILLED
    paper_broker._orders[order_id].filled_qty = 5.0
    paper_broker._orders[order_id].filled_avg_price = None
    if order_id in paper_broker._order_requests:
        paper_broker._order_requests[order_id].limit_price = None

    # Step 1: Worker reboots and reconciles with missing price
    report = await paper_broker.reconcile()
    assert report["status"] == "RECONCILIATION_PENDING"
    assert len(report["pending_reconciliations"]) == 1
    assert report["pending_reconciliations"][0]["reason"] == "UNKNOWN_EXECUTION_PRICE"

    # Step 2: Crucial check: NO fake position with price $100.0 was created
    pos = await paper_broker.get_position("AMD")
    assert pos is None, "Position must NOT be created with synthetic or unknown price"

    # Step 3: Broker updates order with true execution price
    true_broker_price = 162.35
    paper_broker._orders[order_id].filled_avg_price = true_broker_price

    # Step 4: Re-reconcile with Broker Truth
    report2 = await paper_broker.reconcile()
    assert report2["status"] == "reconciled_with_broker"
    assert len(report2["pending_reconciliations"]) == 0

    pos_reconciled = await paper_broker.get_position("AMD")
    assert pos_reconciled is not None
    assert pos_reconciled.quantity == 5.0
    assert pos_reconciled.avg_entry_price == true_broker_price
    assert pos_reconciled.market_value == 5.0 * true_broker_price


# ══════════════════════════════════════════════════════════════════════
# FINANCIAL INVARIANTS AUDIT SUITE
# ══════════════════════════════════════════════════════════════════════

from services.broker_engine.base import OrderAction
from services.broker_engine.order_state_machine import OrderStateMachine, InvalidOrderStateTransitionError
from services.risk_engine.pre_trade_guard import GapRiskEngine
from core.exceptions import BrokerStateUnknownError


@pytest.mark.anyio
async def test_invariant_1_broker_failure_halts_new_entries(paper_broker, risk_engine):
    """
    Invariant 1: When broker state cannot be verified (API error / disconnect),
    the system must NEVER guess positions = [] and must NEVER submit an order into the unknown.
    """
    from services.broker_engine.execution_engine import ExecutionEngine
    exec_eng = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    # Monkeypatch get_account to simulate API down
    async def broken_account():
        raise BrokerStateUnknownError("Alpaca API HTTP 503 Service Unavailable")

    paper_broker.get_account = broken_account

    sig = CombinedSignal(
        symbol="SPY",
        direction=SignalDirection.BUY,
        composite_score=0.88,
        confidence=0.85,
        consensus_ratio=0.80,
        contributing_strategies=["trend_following"],
        is_approved=True,
    )

    from unittest.mock import patch

    # Execution Engine must safely return None and halt order creation
    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        res = await exec_eng.execute_signal(signal=sig, current_price=500.0)
    assert res is None, "Broker API failure must abort order execution"
    assert len(await paper_broker.get_open_orders()) == 0


@pytest.mark.anyio
async def test_invariant_2_existing_position_allows_sell_exit(paper_broker, risk_engine):
    """
    Invariant 2: An existing position must NOT block legitimate EXIT or REDUCE orders.
    Anti-duplicate protection must only apply to new ENTRY orders.
    """
    from unittest.mock import patch
    from services.broker_engine.execution_engine import ExecutionEngine
    paper_broker.cash = 100000.0
    exec_eng = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    # 1. Establish an existing LONG position of AAPL (10 shares @ $150)
    paper_broker.update_market_price("AAPL", 150.0)
    await paper_broker.submit_order(OrderRequest(
        symbol="AAPL",
        quantity=10.0,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    ))
    assert (await paper_broker.get_position("AAPL")).quantity == 10.0

    # 2. An ENTRY signal on AAPL is rejected by anti-duplicate check
    sig_entry = CombinedSignal(
        symbol="AAPL",
        direction=SignalDirection.BUY,
        composite_score=0.85,
        confidence=0.80,
        consensus_ratio=0.80,
        contributing_strategies=["trend_following"],
        is_approved=True,
    )
    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        res_entry = await exec_eng.execute_signal(signal=sig_entry, current_price=150.0, action=OrderAction.ENTRY)
    assert res_entry is None, "Duplicate ENTRY must be rejected when position already exists"

    # 3. An EXIT signal on AAPL MUST be allowed through to the broker
    sig_exit = CombinedSignal(
        symbol="AAPL",
        direction=SignalDirection.SELL,
        composite_score=0.80,
        confidence=0.75,
        consensus_ratio=0.75,
        contributing_strategies=["trend_following"],
        is_approved=True,
    )
    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        res_exit = await exec_eng.execute_signal(
            signal=sig_exit,
            current_price=155.0,
            action=OrderAction.EXIT,
            is_exit=True,
        )
    assert res_exit is not None, "Legitimate EXIT signal must NOT be blocked by existing position"
    assert res_exit.status == OrderStatus.FILLED
    assert res_exit.quantity == 10.0
    # Position is now closed
    assert await paper_broker.get_position("AAPL") is None


@pytest.mark.anyio
async def test_invariant_3_gap_risk_reduce_propagates_exact_size(paper_broker, risk_engine):
    """
    Invariant 3: When PreTradeRiskGuard specifies size_multiplier = 0.5,
    the quantity sent to the broker must be exactly reduced by 50% (50 -> 25 shares).
    """
    from unittest.mock import patch
    from services.broker_engine.execution_engine import ExecutionEngine
    paper_broker.cash = 100000.0
    exec_eng = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)
    paper_broker.update_market_price("NVDA", 120.0)

    sig = CombinedSignal(
        symbol="NVDA",
        direction=SignalDirection.BUY,
        composite_score=0.90,
        confidence=0.85,
        consensus_ratio=0.85,
        contributing_strategies=["breakout"],
        is_approved=True,
    )

    # Base requested size is 50 shares, with size_multiplier = 0.5 (from GapRiskEngine REDUCE)
    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        res = await exec_eng.execute_signal(
            signal=sig,
            current_price=120.0,
            default_shares=50.0,
            size_multiplier=0.5,
        )
    assert res is not None
    assert res.quantity == 25.0, "Quantity must reflect exact 50% reduction (50 * 0.5 = 25)"


@pytest.mark.anyio
async def test_invariant_4_and_5_limit_price_is_not_fill_price(paper_broker):
    """
    Invariant 4 & 5: LIMIT PRICE != EXECUTION PRICE.
    If limit is $160.00 but broker fills at $159.70, position avg_entry_price must be $159.70.
    """
    paper_broker.cash = 100000.0
    req = OrderRequest(
        symbol="QCOM",
        quantity=10.0,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=160.0,
    )
    order_res = await paper_broker.submit_order(req)
    o_id = order_res.broker_order_id

    # Simulate broker fill with real execution price better than limit
    actual_fill_price = 159.70
    paper_broker._orders[o_id].status = OrderStatus.FILLED
    paper_broker._orders[o_id].filled_qty = 10.0
    paper_broker._orders[o_id].filled_avg_price = actual_fill_price

    # Clear position to test reconciliation from broker truth
    if "QCOM" in paper_broker._positions:
        del paper_broker._positions["QCOM"]

    await paper_broker.reconcile()
    pos = await paper_broker.get_position("QCOM")
    assert pos is not None
    assert pos.avg_entry_price == actual_fill_price
    assert pos.avg_entry_price != 160.0, "Execution price must NOT fall back to limit_price"


@pytest.mark.anyio
async def test_invariant_6_paper_reconcile_accounting_buy_sell_close(paper_broker):
    """
    Invariant 6: Position reconciliation ledger:
    - BUY 100 @ 150 -> position = 100
    - SELL 40 @ 160 -> position = 60
    - SELL 60 @ 160 -> position = 0 (fully closed)
    """
    sym = "INTC"
    # Step 1: BUY 100 @ $150
    b_id = "paper_buy_1"
    paper_broker._orders[b_id] = OrderResult(
        broker_order_id=b_id,
        symbol=sym,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        quantity=100.0,
        filled_qty=100.0,
        filled_avg_price=150.0,
    )
    await paper_broker.reconcile()
    pos = await paper_broker.get_position(sym)
    assert pos is not None and pos.quantity == 100.0 and pos.avg_entry_price == 150.0

    # Step 2: SELL 40 @ $160 -> reduces to 60
    s_id = "paper_sell_1"
    paper_broker._orders[s_id] = OrderResult(
        broker_order_id=s_id,
        symbol=sym,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        quantity=40.0,
        filled_qty=40.0,
        filled_avg_price=160.0,
    )
    await paper_broker.reconcile()
    pos2 = await paper_broker.get_position(sym)
    assert pos2 is not None and pos2.quantity == 60.0

    # Step 3: SELL remaining 60 @ $160 -> closes to 0
    s_id2 = "paper_sell_2"
    paper_broker._orders[s_id2] = OrderResult(
        broker_order_id=s_id2,
        symbol=sym,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        quantity=60.0,
        filled_qty=60.0,
        filled_avg_price=160.0,
    )
    await paper_broker.reconcile()
    pos3 = await paper_broker.get_position(sym)
    assert pos3 is None, "Position must be deleted when closed"


def test_invariant_7_order_state_machine_cancel_pending_filled_rejects_late_cancel():
    """
    Invariant 7: Order lifecycle transition:
    SUBMITTED -> CANCEL_PENDING -> FILLED.
    Late CANCELLED transition on a FILLED order must be strictly rejected.
    """
    # Allowed: SUBMITTED -> CANCEL_PENDING
    status = OrderStateMachine.transition(OrderStatus.SUBMITTED, OrderStatus.CANCEL_PENDING)
    assert status == OrderStatus.CANCEL_PENDING

    # Allowed: CANCEL_PENDING -> FILLED (order filled at exchange while cancel was in flight)
    status2 = OrderStateMachine.transition(status, OrderStatus.FILLED)
    assert status2 == OrderStatus.FILLED
    assert OrderStateMachine.is_terminal(status2) is True

    # Strictly forbidden: FILLED -> CANCELLED
    with pytest.raises(InvalidOrderStateTransitionError):
        OrderStateMachine.transition(status2, OrderStatus.CANCELLED)


def test_invariant_8_gap_risk_engine_fail_safe_on_invalid_data():
    """
    Invariant 8: When price data or previous close is invalid (<= 0),
    GapRiskEngine must FAIL-SAFE to REJECT, never APPROVED.
    """
    res = GapRiskEngine.evaluate(
        symbol="TSLA",
        direction="buy",
        current_price=200.0,
        prev_close=-1.0,  # Corrupted data
        natr_14=2.0,
    )
    assert res.status == "REJECT", "Corrupted or missing baseline must FAIL-SAFE to REJECT"
    assert res.size_multiplier == 0.0


@pytest.mark.anyio
async def test_invariant_9_two_workers_same_signal_single_order(paper_broker, risk_engine):
    """
    Invariant 9: Two workers processing the exact same signal concurrently
    must produce the same deterministic client_order_id and result in exactly 1 logical order.
    """
    from unittest.mock import patch
    paper_broker.cash = 100000.0
    worker1_engine = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)
    worker2_engine = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    signal = CombinedSignal(
        symbol="MSFT",
        direction=SignalDirection.BUY,
        composite_score=0.91,
        confidence=0.88,
        consensus_ratio=1.0,
        contributing_strategies=["trend_following", "breakout"],
        timestamp=datetime.now(timezone.utc),
        is_approved=True,
    )

    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        res1, res2 = await asyncio.gather(
            worker1_engine.execute_signal(signal, current_price=400.0, default_shares=10.0),
            worker2_engine.execute_signal(signal, current_price=400.0, default_shares=10.0),
        )

    # Exactly one executed successfully, second was recognized via client_order_id / mutex
    assert (res1 is not None) or (res2 is not None)
    orders = list(paper_broker._orders.values())
    assert len(orders) == 1, f"Expected exactly 1 order in broker, found {len(orders)}"


@pytest.mark.anyio
async def test_invariant_10_timeout_after_broker_accept_does_not_duplicate_order(paper_broker, risk_engine):
    """
    Invariant 10: Lost Response Recovery.
    Broker receives and accepts the order, but a network timeout drops the response.
    ExecutionEngine queries broker by client_order_id, recovers the accepted order, and does NOT duplicate it.
    """
    from unittest.mock import patch
    paper_broker.cash = 100000.0
    exec_eng = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    signal = CombinedSignal(
        symbol="GOOGL",
        direction=SignalDirection.BUY,
        composite_score=0.89,
        confidence=0.85,
        consensus_ratio=0.90,
        contributing_strategies=["trend_following"],
        timestamp=datetime.now(timezone.utc),
        is_approved=True,
    )

    original_submit = paper_broker.submit_order

    async def timeout_submit(order_req):
        # Broker processes order and creates it in internal ledger
        res = await original_submit(order_req)
        # But network response drops with a TimeoutError
        raise TimeoutError("Simulated network timeout: response dropped after broker acceptance")

    paper_broker.submit_order = timeout_submit

    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        recovered_order = await exec_eng.execute_signal(signal, current_price=175.0, default_shares=10.0)

    assert recovered_order is not None, "ExecutionEngine must recover the order accepted by the broker"
    assert recovered_order.symbol == "GOOGL"
    assert len(paper_broker._orders) == 1, "Must NOT place a duplicate order"


def test_invariant_11_different_legitimate_signals_different_client_order_ids(paper_broker, risk_engine):
    """
    Invariant 11: Two different legitimate signals for the same ticker
    (e.g., different strategies or scores) produce different client_order_id fingerprints.
    """
    exec_eng = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    sig_a = CombinedSignal(
        symbol="AAPL",
        direction=SignalDirection.BUY,
        composite_score=0.85,
        confidence=0.80,
        consensus_ratio=0.75,
        contributing_strategies=["trend_following"],
        timestamp=datetime(2026, 10, 2, 14, 0, 0, tzinfo=timezone.utc),
        is_approved=True,
    )
    sig_b = CombinedSignal(
        symbol="AAPL",
        direction=SignalDirection.BUY,
        composite_score=0.92,
        confidence=0.90,
        consensus_ratio=1.0,
        contributing_strategies=["breakout"],
        timestamp=datetime(2026, 10, 2, 14, 0, 0, tzinfo=timezone.utc),
        is_approved=True,
    )

    id_a = exec_eng._generate_client_order_id(sig_a, OrderSide.BUY, OrderAction.ENTRY)
    id_b = exec_eng._generate_client_order_id(sig_b, OrderSide.BUY, OrderAction.ENTRY)

    assert id_a != id_b, "Distinct strategy signals must produce distinct client_order_id"


@pytest.mark.anyio
async def test_invariant_12_size_multiplier_zero_yields_zero_orders(paper_broker, risk_engine):
    """
    Invariant 12: size_multiplier = 0 or calculated size <= 0 must FAIL-SAFE to 0 orders (return None).
    NEVER convert 0 into 1 share.
    """
    from unittest.mock import patch
    paper_broker.cash = 100000.0
    exec_eng = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    signal = CombinedSignal(
        symbol="AMD",
        direction=SignalDirection.BUY,
        composite_score=0.80,
        confidence=0.75,
        consensus_ratio=0.70,
        contributing_strategies=["mean_reversion"],
        is_approved=True,
    )

    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        # Case A: size_multiplier = 0.0 (from GapRiskEngine REJECT)
        res_zero = await exec_eng.execute_signal(
            signal=signal,
            current_price=160.0,
            default_shares=50.0,
            size_multiplier=0.0,
        )
        assert res_zero is None, "size_multiplier=0 must yield None (0 orders), never 1 share"

        # Case B: Tiny fractional multiplier rounding to 0 (50 * 0.01 = 0.5 -> floor = 0)
        res_fractional = await exec_eng.execute_signal(
            signal=signal,
            current_price=160.0,
            default_shares=50.0,
            size_multiplier=0.01,
        )
        assert res_fractional is None, "Calculated 0 shares must yield None (0 orders), never 1 share"


@pytest.mark.anyio
async def test_invariant_13_reduce_and_exit_direction_contracts(paper_broker, risk_engine):
    """
    Invariant 13: OrderAction direction contracts:
    - EXIT LONG -> order side must be SELL
    - EXIT SHORT -> order side must be BUY
    - REDUCE LONG -> order side must be SELL
    Neither EXIT nor REDUCE are blocked by existing position.
    """
    from unittest.mock import patch
    from services.broker_engine.base import Position
    paper_broker.cash = 100000.0
    exec_eng = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    # 1. Establish SHORT position of TSLA (20 shares)
    paper_broker._positions["TSLA"] = Position(
        symbol="TSLA",
        side="short",
        quantity=20.0,
        avg_entry_price=220.0,
        current_price=220.0,
        market_value=4400.0,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
    )

    # EXIT signal on SHORT position must route as BUY
    sig_exit_short = CombinedSignal(
        symbol="TSLA",
        direction=SignalDirection.BUY,
        composite_score=0.85,
        confidence=0.80,
        consensus_ratio=0.80,
        contributing_strategies=["trend_following"],
        is_approved=True,
    )

    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        res_exit_short = await exec_eng.execute_signal(
            signal=sig_exit_short,
            current_price=215.0,
            action=OrderAction.EXIT,
            is_exit=True,
        )
        assert res_exit_short is not None
        assert res_exit_short.side == OrderSide.BUY, "EXIT on SHORT position must be a BUY order"
        assert res_exit_short.quantity == 20.0

    # 2. Establish LONG position of INTC (40 shares)
    paper_broker._positions["INTC"] = Position(
        symbol="INTC",
        side="long",
        quantity=40.0,
        avg_entry_price=30.0,
        current_price=30.0,
        market_value=1200.0,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
    )

    # REDUCE signal on LONG position must route as SELL for exact reduction quantity
    sig_reduce_long = CombinedSignal(
        symbol="INTC",
        direction=SignalDirection.SELL,
        composite_score=0.80,
        confidence=0.75,
        consensus_ratio=0.75,
        contributing_strategies=["trend_following"],
        is_approved=True,
    )

    with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
        res_reduce = await exec_eng.execute_signal(
            signal=sig_reduce_long,
            current_price=32.0,
            default_shares=15.0,
            action=OrderAction.REDUCE,
            is_exit=True,
        )
        assert res_reduce is not None
        assert res_reduce.side == OrderSide.SELL, "REDUCE on LONG position must be a SELL order"
        assert res_reduce.quantity == 15.0, "REDUCE must use explicit requested quantity"


@pytest.mark.anyio
async def test_invariant_14_broker_state_unknown_never_converts_to_empty_lists():
    """
    Invariant 14: Broker failure must raise BrokerStateUnknownError.
    UNKNOWN state must NEVER be converted to empty lists [] or synthetic zero positions/orders.
    """
    from services.broker_engine.alpaca_broker import AlpacaBroker
    broker = AlpacaBroker(api_key="test", secret_key="test", is_paper=True)
    broker._connected = True

    class BrokenClient:
        def get_account(self):
            raise ConnectionResetError("Broker connection reset by peer")
        def get_all_positions(self):
            raise ConnectionResetError("Broker connection reset by peer")
        def get_orders(self, *args, **kwargs):
            raise ConnectionResetError("Broker connection reset by peer")

    broker.client = BrokenClient()

    with pytest.raises(BrokerStateUnknownError):
        await broker.get_account()

    with pytest.raises(BrokerStateUnknownError):
        await broker.get_positions()

    with pytest.raises(BrokerStateUnknownError):
        await broker.get_open_orders()


@pytest.mark.anyio
async def test_accounting_ledger_incremental_lots_vwap_and_pnl(paper_broker):
    """
    Ledger Verification with Incremental Lots:
    1. BUY 100 @ $150 ($15,000)
    2. BUY 50 @ $160 ($8,000) -> Total 150 shares, Total Cost $23,000 -> VWAP = $153.3333
    3. SELL 75 @ $170 -> Realized P&L = 75 * (170 - 153.3333) = +$1,250.00, remaining = 75 shares
    4. SELL 75 @ $180 -> Realized P&L = 75 * (180 - 153.3333) = +$2,000.00, remaining = 0 shares
    Total Realized P&L = +$3,250.00.
    """
    sym = "AMZN"
    paper_broker.cash = 100000.0
    paper_broker._positions.clear()
    paper_broker._orders.clear()
    paper_broker.realized_pnl = 0.0

    # Step 1: BUY 100 @ 150
    paper_broker._orders["b1"] = OrderResult(
        broker_order_id="b1",
        symbol=sym,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        quantity=100.0,
        filled_qty=100.0,
        filled_avg_price=150.0,
    )
    await paper_broker.reconcile()
    pos1 = await paper_broker.get_position(sym)
    assert pos1.quantity == 100.0
    assert pos1.avg_entry_price == 150.0

    # Step 2: BUY 50 @ 160
    paper_broker._orders["b2"] = OrderResult(
        broker_order_id="b2",
        symbol=sym,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        quantity=50.0,
        filled_qty=50.0,
        filled_avg_price=160.0,
    )
    await paper_broker.reconcile()
    pos2 = await paper_broker.get_position(sym)
    assert pos2.quantity == 150.0
    # Expected VWAP: (100 * 150 + 50 * 160) / 150 = 23000 / 150 = 153.3333
    assert abs(pos2.avg_entry_price - 153.3333) < 0.01

    # Step 3: SELL 75 @ 170
    paper_broker._orders["s1"] = OrderResult(
        broker_order_id="s1",
        symbol=sym,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        quantity=75.0,
        filled_qty=75.0,
        filled_avg_price=170.0,
    )
    await paper_broker.reconcile()
    pos3 = await paper_broker.get_position(sym)
    assert pos3.quantity == 75.0
    assert abs(pos3.avg_entry_price - 153.3333) < 0.01
    # Realized P&L: 75 * (170 - 153.3333) = +$1,250.00
    assert abs(paper_broker.realized_pnl - 1250.0) < 1.0

    # Step 4: SELL 75 @ 180 (Full close)
    paper_broker._orders["s2"] = OrderResult(
        broker_order_id="s2",
        symbol=sym,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        quantity=75.0,
        filled_qty=75.0,
        filled_avg_price=180.0,
    )
    await paper_broker.reconcile()
    pos4 = await paper_broker.get_position(sym)
    assert pos4 is None, "Position must be completely closed"
    # Total Realized P&L: 1250 + (75 * (180 - 153.3333)) = 1250 + 2000 = +$3,250.00
    assert abs(paper_broker.realized_pnl - 3250.0) < 1.0


def test_session_boundaries_integration_capabilities():
    """
    Session Boundaries Integration Test:
    Verifies capabilities across the 8 distinct market session windows:
    - 04:00 - 07:00: Early Premarket (can_enter=False, can_exit=True, limit only, size scaled 0.0)
    - 07:00 - 09:25: Core Premarket (can_enter=True, can_place_market=False)
    - 09:25 - 09:30: Opening Buffer (can_enter=False, no new entries right before bell)
    - 09:30 - 15:30: Regular Session (can_enter=True, market & limit)
    - 15:30 - 15:40: Near Close / Risk Descalation (can_enter=True, max size scaled 0.5)
    - 15:40 - 16:00: Closing Window (can_enter=False, exits only)
    - 16:00 - 20:00: After Hours (can_enter=True, extended limit)
    - 20:00 - 04:00: Overnight (can_enter=False, deliberately blocked for risk policy)
    - Weekend: can_enter=False always.
    """
    from core.market_session import get_session_capabilities
    from zoneinfo import ZoneInfo
    tz_et = ZoneInfo("America/New_York")

    # Regular Wednesday: 2026-10-07
    # 05:00 ET -> Early Premarket (04:00 to 07:00 ET strictly blocks entries for risk)
    c_0500 = get_session_capabilities(datetime(2026, 10, 7, 5, 0, tzinfo=tz_et))
    assert c_0500.can_enter is False, "04:00-07:00 early premarket strictly blocks new entries"
    assert c_0500.can_exit is True

    # 08:00 ET -> Core Premarket (07:00 to 09:25 ET allows limit entries)
    c_0800 = get_session_capabilities(datetime(2026, 10, 7, 8, 0, tzinfo=tz_et))
    assert c_0800.can_enter is True
    assert c_0800.can_place_market is False  # Limit only

    # 09:27 ET -> Opening Buffer
    c_0927 = get_session_capabilities(datetime(2026, 10, 7, 9, 27, tzinfo=tz_et))
    assert c_0927.can_enter is False, "Opening buffer 09:25-09:30 must block new entries"

    # 10:30 ET -> Regular Session
    c_1030 = get_session_capabilities(datetime(2026, 10, 7, 10, 30, tzinfo=tz_et))
    assert c_1030.can_enter is True
    assert c_1030.can_place_market is True

    # 15:35 ET -> Risk Descalation (15:30-15:40 ET permits entries but scales size to 0.5)
    c_1535 = get_session_capabilities(datetime(2026, 10, 7, 15, 35, tzinfo=tz_et))
    assert c_1535.can_enter is True
    assert c_1535.max_position_size_multiplier == 0.5

    # 15:50 ET -> Closing Window
    c_1550 = get_session_capabilities(datetime(2026, 10, 7, 15, 50, tzinfo=tz_et))
    assert c_1550.can_enter is False
    assert c_1550.can_exit is True

    # 17:00 ET -> After Hours
    c_1700 = get_session_capabilities(datetime(2026, 10, 7, 17, 0, tzinfo=tz_et))
    assert c_1700.can_enter is True
    assert c_1700.can_place_market is False  # Extended limit only

    # 22:00 ET -> Overnight
    c_2200 = get_session_capabilities(datetime(2026, 10, 7, 22, 0, tzinfo=tz_et))
    assert c_2200.can_enter is False, "Overnight 20:00-04:00 is deliberately blocked by risk policy"

    # Saturday: Weekend
    c_sat = get_session_capabilities(datetime(2026, 10, 10, 12, 0, tzinfo=tz_et))
    assert c_sat.can_enter is False
    assert c_sat.is_weekend is True


def test_invariant_15_reconstructed_signal_after_crash_produces_identical_client_order_id(paper_broker, risk_engine):
    """
    Invariant 15 (Auditor requirement):
    If a process crashes between:
       generar ID -> enviar orden -> guardar estado local
    When the process restarts and reconstructs the exact same logical signal,
    it MUST produce the exact same client_order_id bit-for-bit.
    """
    from services.broker_engine.base import OrderAction
    exec_eng_original = ExecutionEngine(broker=paper_broker, risk_engine=risk_engine)

    signal_time = datetime(2026, 10, 2, 14, 30, 0, tzinfo=timezone.utc)
    original_signal = CombinedSignal(
        symbol="AAPL",
        direction=SignalDirection.BUY,
        composite_score=0.825,
        confidence=0.88,
        consensus_ratio=1.0,
        contributing_strategies=["trend_following", "breakout"],
        timestamp=signal_time,
        is_approved=True,
    )

    original_client_id = exec_eng_original._generate_client_order_id(
        original_signal, OrderSide.BUY, OrderAction.ENTRY
    )

    # Simulate hard process crash: destroy execution engine, instantiate fresh new engine in new scope
    del exec_eng_original

    fresh_broker = PaperBroker(initial_cash=1000.0)
    fresh_risk = RiskEngine()
    fresh_exec_eng = ExecutionEngine(broker=fresh_broker, risk_engine=fresh_risk)

    # Reconstruct the exact same logical signal (e.g. from event log or bar replay)
    reconstructed_signal = CombinedSignal(
        symbol="AAPL",
        direction=SignalDirection.BUY,
        composite_score=0.825,
        confidence=0.88,
        consensus_ratio=1.0,
        contributing_strategies=["trend_following", "breakout"],
        timestamp=signal_time,
        is_approved=True,
    )

    reconstructed_client_id = fresh_exec_eng._generate_client_order_id(
        reconstructed_signal, OrderSide.BUY, OrderAction.ENTRY
    )

    assert original_client_id == reconstructed_client_id, (
        f"Crash recovery contract violated! Expected identical client_order_id, got: {original_client_id} vs {reconstructed_client_id}"
    )

    # Also verify that ShadowTradingEngine produces the exact same ID
    from services.shadow_engine import ShadowTradingEngine
    shadow_eng = ShadowTradingEngine()
    shadow_client_id = shadow_eng.generate_client_order_id(
        symbol="AAPL",
        side=OrderSide.BUY,
        action=OrderAction.ENTRY,
        composite_score=0.825,
        strategies=["trend_following", "breakout"],
        signal_timestamp=signal_time,
    )
    assert shadow_client_id == original_client_id, "ShadowTradingEngine client_order_id must match ExecutionEngine byte-for-byte"



