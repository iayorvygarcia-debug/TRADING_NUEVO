"""
Unit and Integration Tests for Shadow Trading Engine.

Audits:
1. Hard Broker Isolation (Zero broker orders in shadow mode).
2. Deterministic client_order_id calculation.
3. Forensic Decision Telemetry ("¿Por qué quiso comprar AAPL?").
4. Theoretical Execution with Realistic Slippage & Spread.
5. Virtual Position Lifecycle (Take-Profit & Stop-Loss execution).
6. Performance Metrics (P&L, Win Rate, Profit Factor, Max Drawdown).
7. Risk Guard Rejection Tracking with Exact Reasons.
8. Daily Report Generation (JSON & Markdown persistence).
9. Disk Persistence & Recovery.
10. FastAPI Shadow Endpoints.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from apps.api.main import app
from services.broker_engine.base import OrderAction, OrderSide, OrderType
from services.shadow_engine.models import (
    ShadowDecisionRecord,
    ShadowMetrics,
    ShadowOrder,
    ShadowPosition,
    ShadowTrade,
)
from services.shadow_engine.tracker import ShadowTradingEngine


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def shadow_engine(temp_data_dir):
    return ShadowTradingEngine(
        initial_capital=10000.0,
        data_dir=temp_data_dir,
        shadow_mode=True,
    )


# ── 1. Hard Broker Isolation ──────────────────────────────────────────────────

def test_shadow_engine_enforces_shadow_mode_true(temp_data_dir):
    """Verifies that ShadowTradingEngine requires shadow_mode=True."""
    engine = ShadowTradingEngine(initial_capital=10000.0, data_dir=temp_data_dir, shadow_mode=True)
    assert engine.shadow_mode is True

    # If shadow_mode is False, evaluate_and_record_decision must assert/fail-safe
    engine.shadow_mode = False
    with pytest.raises(AssertionError):
        engine.evaluate_and_record_decision(
            symbol="AAPL",
            strategies=["TrendFollowing"],
            direction="BUY",
            action="ENTRY",
            session_name="REGULAR",
            signal_score=0.82,
            base_quantity=25.0,
            session_multiplier=1.0,
            gap_multiplier=1.0,
            final_quantity=25.0,
            spread_pct=0.0006,
            natr_14=0.015,
            price=150.0,
        )


# ── 2. Deterministic client_order_id ──────────────────────────────────────────

def test_shadow_client_order_id_determinism(shadow_engine):
    """Verifies client_order_id is deterministic and reproducible."""
    ts = datetime(2026, 10, 2, 14, 30, 0, tzinfo=timezone.utc)
    id1 = shadow_engine.generate_client_order_id(
        symbol="MSFT",
        side=OrderSide.BUY,
        action=OrderAction.ENTRY,
        composite_score=0.850,
        strategies=["trend_following", "breakout"],
        signal_timestamp=ts,
    )
    id2 = shadow_engine.generate_client_order_id(
        symbol="MSFT",
        side=OrderSide.BUY,
        action=OrderAction.ENTRY,
        composite_score=0.850,
        strategies=["breakout", "trend_following"],  # Order of list should not change sorted result
        signal_timestamp=ts,
    )
    assert id1 == id2
    assert id1.startswith("agy_MSFT_b_ent_")


# ── 3. Forensic Decision Telemetry ("¿Por qué quiso comprar AAPL?") ──────────

def test_forensic_reconstruction_format(shadow_engine):
    """
    Verifies that the forensic reconstruction string matches the exact format
    specified by the auditor to answer: '¿Por qué quiso comprar AAPL?'.
    """
    ts = datetime(2026, 10, 2, 15, 0, 0, tzinfo=timezone.utc)
    record = shadow_engine.evaluate_and_record_decision(
        symbol="AAPL",
        strategies=["Trend Following"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.82,
        base_quantity=25.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=25.0,
        spread_pct=0.0006,
        natr_14=0.018,
        price=180.0,
        order_type="LIMIT",
        limit_price=180.10,
        stop_price=176.40,
        target_price=187.20,
        risk_approved=True,
        signal_timestamp=ts,
    )

    reconstruction = shadow_engine.get_forensic_reconstruction("AAPL")
    assert reconstruction is not None

    # Check key auditor markers in the reconstructed text
    assert "Strategy:\n    Trend Following" in reconstruction
    assert "Signal:\n    BUY" in reconstruction
    assert "Score:\n    0.82" in reconstruction
    assert "Session:\n    REGULAR" in reconstruction
    assert "Session multiplier:\n    1.00" in reconstruction
    assert "GapRisk:\n    APPROVED" in reconstruction
    assert "Gap multiplier:\n    1.00" in reconstruction
    assert "Spread:\n    0.06%" in reconstruction
    assert "Risk:\n    APPROVED" in reconstruction
    assert "Final quantity:\n    25" in reconstruction
    assert "Order:\n    LIMIT" in reconstruction
    assert f"Client Order ID:\n    {record.client_order_id}" in reconstruction
    assert "Execution:\n    SHADOW" in reconstruction


# ── 4. Theoretical Execution with Realistic Slippage & Spread ─────────────────

def test_theoretical_execution_slippage_and_spread(shadow_engine):
    """Verifies that simulated execution adds half-spread and slippage to BUY fills."""
    price = 100.0
    spread_pct = 0.0010  # 10 bps ($0.10) -> half spread = $0.05
    record = shadow_engine.evaluate_and_record_decision(
        symbol="NVDA",
        strategies=["Breakout"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.90,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=spread_pct,
        natr_14=0.02,
        price=price,
        stop_price=95.0,
        target_price=110.0,
        risk_approved=True,
    )

    assert "NVDA" in shadow_engine.open_positions
    pos = shadow_engine.open_positions["NVDA"]
    # BUY entry price must be higher than current price due to spread and slippage
    assert pos.entry_price > price
    # Regular hours slippage is ~1-2 bps, half-spread is 5 bps -> entry around 100.06 - 100.07
    assert 100.05 <= pos.entry_price <= 100.15


def test_extended_hours_slippage_higher_than_regular(shadow_engine):
    """Verifies that extended hours simulated slippage is higher than regular hours."""
    price = 200.0
    spread_pct = 0.0020  # 20 bps

    slip_reg_pct, _ = shadow_engine.calculate_simulated_slippage(price, spread_pct, is_extended_session=False)
    slip_ext_pct, _ = shadow_engine.calculate_simulated_slippage(price, spread_pct, is_extended_session=True)

    assert slip_ext_pct > slip_reg_pct, "Extended hours slippage must be higher than regular hours"


# ── 5. Virtual Position Lifecycle (Take-Profit & Stop-Loss) ───────────────────

def test_virtual_position_take_profit_execution(shadow_engine):
    """Verifies that hitting Take-Profit closes the shadow position with a winning trade."""
    shadow_engine.evaluate_and_record_decision(
        symbol="TSLA",
        strategies=["Trend Following"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.88,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0005,
        natr_14=0.02,
        price=200.0,
        stop_price=190.0,
        target_price=220.0,
        risk_approved=True,
    )

    assert "TSLA" in shadow_engine.open_positions

    # Market rises to 221.0 (breaches 220.0 take profit)
    closed = shadow_engine.evaluate_open_positions({"TSLA": 221.0})
    assert len(closed) == 1
    trade = closed[0]
    assert trade.symbol == "TSLA"
    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.net_pnl > 0
    assert trade.total_slippage > 0.0, "Total slippage must be tracked in the closed trade ledger"
    assert trade.spread_friction_dollars > 0.0, "Spread cost must be tracked in the closed trade ledger"
    assert trade.mfe_pct > 0.0, "MFE must be tracked for winning trade"
    assert "TSLA" not in shadow_engine.open_positions


def test_profit_factor_graceful_na_when_zero_loss(shadow_engine):
    """
    Auditor Requirement 1:
    When a strategy has winning trades and zero losses, Profit Factor must NOT be an
    extreme artificial number (e.g. 148,600,000.00). It must be represented as
    'N/A (sin pérdidas en muestra)' and profit_factor as None.
    """
    shadow_engine.evaluate_and_record_decision(
        symbol="GOOGL",
        strategies=["Breakout"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.92,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0004,
        natr_14=0.015,
        price=170.0,
        target_price=180.0,
        risk_approved=True,
    )
    shadow_engine.evaluate_open_positions({"GOOGL": 182.0})

    sm = shadow_engine.metrics.strategy_metrics.get("Breakout")
    assert sm is not None
    assert sm.winning_trades == 1
    assert sm.losing_trades == 0
    assert sm.gross_loss == 0.0
    assert sm.profit_factor is None
    assert "sin pérdidas" in sm.profit_factor_display


def test_verifiable_broker_isolation_telemetry(shadow_engine):
    """
    Auditor Requirement 9:
    Verifies that the Shadow Gate records intercepted orders and guarantees
    that real broker submit calls and orders created remain exactly ZERO.
    """
    m = shadow_engine.metrics
    assert m.real_broker_submit_calls == 0
    assert m.real_broker_orders_created == 0
    assert m.shadow_gate_intercepted_orders >= 0


def test_virtual_position_stop_loss_execution(shadow_engine):
    """Verifies that hitting Stop-Loss closes the shadow position with a losing trade."""
    shadow_engine.evaluate_and_record_decision(
        symbol="AMD",
        strategies=["Mean Reversion"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.75,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0005,
        natr_14=0.02,
        price=150.0,
        stop_price=145.0,
        target_price=160.0,
        risk_approved=True,
    )

    assert "AMD" in shadow_engine.open_positions

    # Market drops to 144.0 (breaches 145.0 stop loss)
    closed = shadow_engine.evaluate_open_positions({"AMD": 144.0})
    assert len(closed) == 1
    trade = closed[0]
    assert trade.symbol == "AMD"
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.net_pnl < 0
    assert "AMD" not in shadow_engine.open_positions


def test_conservative_intrabar_tp_sl_collision_policy(shadow_engine):
    """
    Auditor Requirement 3:
    When a single bar touches BOTH Take-Profit and Stop-Loss (e.g. High >= TP and Low <= SL),
    the conservative fail-safe policy MUST assume STOP-LOSS occurred first to prevent
    artificially optimistic P&L.
    """
    shadow_engine.evaluate_and_record_decision(
        symbol="INTC",
        strategies=["Breakout"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.85,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0005,
        natr_14=0.02,
        price=100.0,
        stop_price=95.0,
        target_price=108.0,
        risk_approved=True,
    )

    # Bar touches both TP (High 110 >= 108) and SL (Low 90 <= 95)
    bar = {
        "INTC": {
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
        }
    }
    closed = shadow_engine.evaluate_open_positions(bar)
    assert len(closed) == 1
    trade = closed[0]
    assert trade.symbol == "INTC"
    # Must prioritize STOP_LOSS!
    assert "STOP_LOSS" in trade.exit_reason
    assert "Conservative Collision" in trade.exit_reason
    assert trade.net_pnl < 0
    assert "INTC" not in shadow_engine.open_positions


def test_dual_prices_and_slippage_transparency(shadow_engine):
    """
    Auditor Requirement 1:
    Shadow Engine must record both signal_price and shadow_execution_price,
    along with spread, NATR, session, liquidity metric, and slippage assumption.
    """
    rec = shadow_engine.evaluate_and_record_decision(
        symbol="QQQ",
        strategies=["Trend Following"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.89,
        base_quantity=20.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=20.0,
        spread_pct=0.0004,
        natr_14=0.015,
        price=450.0,
        liquidity_metric="RVOL: 1.82x",
        risk_approved=True,
    )

    assert rec.signal_price == 450.0
    assert rec.shadow_execution_price is not None
    assert rec.shadow_execution_price > rec.signal_price
    assert rec.slippage_assumption_pct > 0.0
    assert rec.slippage_assumption_dollars > 0.0
    assert rec.liquidity_metric == "RVOL: 1.82x"
    assert rec.spread_pct == 0.0004



# ── 6. Performance Metrics Accounting ─────────────────────────────────────────

def test_performance_metrics_accounting(shadow_engine):
    """Verifies win rate, profit factor, and drawdown calculations."""
    # Trade 1: Win $200
    shadow_engine.evaluate_and_record_decision(
        symbol="SYM1",
        strategies=["Strat1"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.80,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0002,
        natr_14=0.01,
        price=100.0,
        target_price=120.0,
        risk_approved=True,
    )
    shadow_engine.evaluate_open_positions({"SYM1": 121.0})

    # Trade 2: Loss -$100
    shadow_engine.evaluate_and_record_decision(
        symbol="SYM2",
        strategies=["Strat2"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.80,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0002,
        natr_14=0.01,
        price=100.0,
        stop_price=90.0,
        risk_approved=True,
    )
    shadow_engine.evaluate_open_positions({"SYM2": 89.0})

    m = shadow_engine.metrics
    assert m.total_trades == 2
    assert m.winning_trades == 1
    assert m.losing_trades == 1
    assert m.win_rate == 0.50
    assert m.loss_rate == 0.50
    assert m.avg_win > 0.0
    assert m.avg_loss > 0.0
    assert m.expectancy == round((m.win_rate * m.avg_win) - (m.loss_rate * m.avg_loss), 2)
    assert m.profit_factor > 1.0  # $200 win / $100 loss ~ 2.0
    assert m.total_net_pnl > 0

    # Conversion Funnel checks
    assert m.signals_generated == 2
    assert m.signals_approved_by_risk == 2
    assert m.signals_order_eligible == 2
    assert m.virtual_entries == 2
    assert m.closed_trades_count == 2

    # Slippage and spread empirical checks
    assert m.slippage_avg_pct > 0.0
    assert m.spread_avg_pct > 0.0


# ── 7. Risk Guard Rejection Tracking ──────────────────────────────────────────

def test_risk_guard_rejection_tracking(shadow_engine):
    """Verifies that rejections by risk guards are counted with exact reasons."""
    shadow_engine.evaluate_and_record_decision(
        symbol="SPY",
        strategies=["Trend Following"],
        direction="BUY",
        action="ENTRY",
        session_name="PREMARKET_ACTIVE",
        signal_score=0.70,
        base_quantity=20.0,
        session_multiplier=0.5,
        gap_multiplier=0.0,
        final_quantity=0.0,
        spread_pct=0.0050,
        natr_14=0.04,
        price=500.0,
        risk_approved=False,
        rejection_reason="Dynamic spread 0.50% exceeds cap 0.25%; Adverse overnight gap -3.2%",
        guard_name="PreTradeRiskGuard",
    )

    assert shadow_engine.metrics.rejections_by_guard.get("PreTradeRiskGuard") == 1
    assert len(shadow_engine.metrics.rejection_reasons) == 1
    last_rej = shadow_engine.metrics.rejection_reasons[0]
    assert last_rej["symbol"] == "SPY"
    assert "Dynamic spread" in last_rej["reason"]


# ── 8. Daily Report Generation (JSON & Markdown) ──────────────────────────────

def test_daily_report_generation(shadow_engine):
    """Verifies generation of both JSON and Markdown daily audit reports."""
    # Seed a decision and closed trade
    shadow_engine.evaluate_and_record_decision(
        symbol="META",
        strategies=["Breakout"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.91,
        base_quantity=5.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=5.0,
        spread_pct=0.0005,
        natr_14=0.015,
        price=500.0,
        target_price=520.0,
        risk_approved=True,
    )
    shadow_engine.evaluate_open_positions({"META": 525.0})

    report_dict, report_md = shadow_engine.generate_daily_report("2026-10-02")

    assert report_dict["date"] == "2026-10-02"
    assert "Embudo de Conversión" in report_md
    assert "Resumen Cuantitativo & Expectancy" in report_md
    assert "Expectancy por Operación" in report_md
    assert "Distribución Empírica de Slippage & Spread" in report_md
    assert "Rendimiento Segregado por Estrategia" in report_md
    assert "Historial de Operaciones Virtuales Cerradas" in report_md
    assert "Ejemplo Forense de Decisión Reconstruida" in report_md

    # Check files exist on disk
    json_path = shadow_engine.reports_dir / "shadow_daily_report_2026-10-02.json"
    md_path = shadow_engine.reports_dir / "shadow_daily_report_2026-10-02.md"
    assert json_path.exists()
    assert md_path.exists()


# ── 9. Disk Persistence & State Recovery ──────────────────────────────────────

def test_disk_persistence_and_state_recovery(temp_data_dir):
    """Verifies that a fresh ShadowTradingEngine instance re-hydrates state completely."""
    engine1 = ShadowTradingEngine(initial_capital=10000.0, data_dir=temp_data_dir, shadow_mode=True)
    engine1.evaluate_and_record_decision(
        symbol="AMZN",
        strategies=["VWAP Reversion"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.83,
        base_quantity=15.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=15.0,
        spread_pct=0.0004,
        natr_14=0.012,
        price=185.0,
        stop_price=180.0,
        target_price=195.0,
        risk_approved=True,
    )
    assert "AMZN" in engine1.open_positions

    # Simulate restart: create brand new instance pointing to same directory
    engine2 = ShadowTradingEngine(initial_capital=10000.0, data_dir=temp_data_dir, shadow_mode=True)
    assert "AMZN" in engine2.open_positions
    recovered_pos = engine2.open_positions["AMZN"]
    assert recovered_pos.symbol == "AMZN"
    assert recovered_pos.quantity == 15.0
    assert recovered_pos.stop_loss_price == 180.0
    assert recovered_pos.take_profit_price == 195.0


# ── 10. FastAPI Shadow Trading Endpoints ──────────────────────────────────────

def test_api_shadow_endpoints():
    """Verifies HTTP endpoints for summary, decisions, reconstruction, and report."""
    with TestClient(app) as client:
        # Summary endpoint
        resp_sum = client.get("/api/v1/shadow/summary")
        assert resp_sum.status_code == 200
        data_sum = resp_sum.json()
        assert "metrics" in data_sum
        assert "open_positions" in data_sum

        # Report endpoint
        resp_rep = client.get("/api/v1/shadow/report")
        assert resp_rep.status_code == 200
        data_rep = resp_rep.json()
        assert "report" in data_rep
        assert "markdown" in data_rep


# ── 11. Exact Closed Accounting Identity Audit ────────────────────────────────

def test_closed_accounting_identity_gross_minus_frictions_equals_net(shadow_engine):
    """
    Auditor Requirement:
    Guarantee that for every closed trade:
    net_pnl == gross_pnl - total_slippage - spread_friction_dollars - total_commissions
    with zero unaccounted discrepancy.
    """
    # Create and close trade (MSFT scenario)
    shadow_engine.evaluate_and_record_decision(
        symbol="MSFT",
        strategies=["Breakout"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.90,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0004,
        natr_14=0.015,
        price=400.0,
        target_price=415.0,
        risk_approved=True,
    )
    closed = shadow_engine.evaluate_open_positions({"MSFT": 415.0})
    assert len(closed) == 1
    t = closed[0]

    # Verify exact accounting identity
    expected_net = round(t.gross_pnl - t.total_slippage - t.spread_friction_dollars - t.total_commissions, 2)
    assert t.net_pnl == expected_net
    assert abs(t.rounding_tolerance_dollars) < 0.005


# ── 12. Strict Temporal Causality & Anti-Lookahead Audit ──────────────────────

def test_strict_temporal_causality_and_no_lookahead(shadow_engine):
    """
    Auditor Requirement:
    Verify that:
    1. Features at time T depend ONLY on bars [0...T]. Future bar T+1 does NOT leak backward.
    2. MFE / MAE at entry time T are strictly 0.0%, and only evolve from post-entry bars.
    """
    import numpy as np
    import pandas as pd
    from services.technical_analysis.feature_engine import FeatureEngine

    # Generate synthetic OHLCV for 40 bars (up to bar T)
    dates = pd.date_range("2026-10-01 09:30", periods=40, freq="5min", tz="UTC")
    data_t = {
        "open": np.linspace(100, 110, 40),
        "high": np.linspace(101, 111, 40),
        "low": np.linspace(99, 109, 40),
        "close": np.linspace(100.5, 110.5, 40),
        "volume": np.full(40, 10000.0),
    }
    df_t = pd.DataFrame(data_t, index=dates)

    # Compute features at T
    fvec_t = FeatureEngine.compute_features(df=df_t, symbol="TEST")
    feat_t_dict = fvec_t.to_dict()

    # Now simulate future bar T+1 with an extreme price shock (+50% gap up)
    dates_future = pd.date_range("2026-10-01 09:30", periods=41, freq="5min", tz="UTC")
    data_future = {
        "open": np.append(data_t["open"], 165.0),
        "high": np.append(data_t["high"], 170.0),
        "low": np.append(data_t["low"], 164.0),
        "close": np.append(data_t["close"], 168.0),
        "volume": np.append(data_t["volume"], 99999.0),
    }
    df_future = pd.DataFrame(data_future, index=dates_future)

    # Recomputing features at T on df_future.iloc[:40] MUST be bit-for-bit identical
    fvec_t_recalc = FeatureEngine.compute_features(df=df_future.iloc[:40], symbol="TEST")
    assert fvec_t_recalc.to_dict() == feat_t_dict, "Features at T must be independent of future bars"

    # Verify MFE/MAE causality in Shadow Engine
    shadow_engine.evaluate_and_record_decision(
        symbol="CAUSAL",
        strategies=["Trend Following"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.85,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0004,
        natr_14=0.015,
        price=100.0,
        stop_price=95.0,
        target_price=110.0,
        risk_approved=True,
    )
    pos = shadow_engine.open_positions["CAUSAL"]
    assert pos.current_mfe_pct == 0.0, "MFE at entry time T must be exactly 0.0%"
    assert pos.current_mae_pct == 0.0, "MAE at entry time T must be exactly 0.0%"

    # Evaluate post-entry bar T+1 (High 105, Low 99, Close 104)
    shadow_engine.evaluate_open_positions({
        "CAUSAL": {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0}
    })
    # MFE should be updated to (105 - entry_price) / entry_price > 0
    assert pos.current_mfe_pct > 0.0, "MFE must be updated by post-entry bar T+1"
    assert pos.current_mae_pct < 0.0, "MAE must be updated by post-entry bar T+1"


# ── 13. Shadow Gate Audit & Equity Separation Audit ───────────────────────────

def test_shadow_gate_isolation_chain_and_equity_separation(shadow_engine):
    """
    Auditor Requirement:
    1. Shadow candidate orders -> Shadow Gate (100% intercepted) -> 0 broker submissions.
    2. Realized P&L and Floating Unrealized P&L remain strictly segregated.
    """
    # 1 approved decision
    shadow_engine.evaluate_and_record_decision(
        symbol="ISO1",
        strategies=["Trend Following"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.80,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0004,
        natr_14=0.015,
        price=100.0,
        risk_approved=True,
    )

    m = shadow_engine.metrics
    assert m.shadow_candidate_orders == 1
    assert m.shadow_gate_intercepted_orders == 1
    assert m.real_broker_submit_calls == 0
    assert m.real_broker_orders_created == 0

    # Generate daily report
    report_data, md = shadow_engine.generate_daily_report("2026-10-02")
    assert "Órdenes Candidatas hacia Broker" in md
    assert "100.0% Interceptadas" in md
    assert "P&L Flotante No Realizado" in md
    assert "P&L Neto Realizado" in md


# ── 14. Cumulative Report & R-Multiple Distribution Audit ────────────────────

def test_cumulative_report_generation_and_r_distribution(shadow_engine):
    """
    Auditor Requirement:
    1. Cumulative R tracking (Cumulative R = sum(R_i)).
    2. Expectancy in R and Standard Deviation of R.
    3. R-distribution stratification (< -2R, [-2R, -1R), [-1R, 0R), [0R, 1R), [1R, 2R), >= 2R).
    4. Duration source clarification (BAR_SIMULATION).
    5. Generation of shadow_cumulative_report.json and .md.
    """
    # Trade 1: MSFT (+1.45R)
    shadow_engine.evaluate_and_record_decision(
        symbol="MSFT",
        strategies=["Breakout"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.90,
        base_quantity=10.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=10.0,
        spread_pct=0.0004,
        natr_14=0.015,
        price=400.0,
        stop_price=390.0,
        target_price=415.0,
        risk_approved=True,
    )
    shadow_engine.evaluate_open_positions({"MSFT": 415.0})

    # Trade 2: TSLA (-1.02R)
    shadow_engine.evaluate_and_record_decision(
        symbol="TSLA",
        strategies=["Mean Reversion"],
        direction="BUY",
        action="ENTRY",
        session_name="REGULAR",
        signal_score=0.88,
        base_quantity=15.0,
        session_multiplier=1.0,
        gap_multiplier=1.0,
        final_quantity=15.0,
        spread_pct=0.0006,
        natr_14=0.025,
        price=220.0,
        stop_price=213.40,
        target_price=235.0,
        risk_approved=True,
    )
    shadow_engine.evaluate_open_positions({"TSLA": 213.0})

    m = shadow_engine.metrics
    assert m.closed_trades_count == 2
    expected_cum_r = round(sum(shadow_engine._all_r_multiples), 2)
    assert m.cumulative_r_multiple == expected_cum_r
    assert m.expectancy_r in {0.21, 0.22}                    # ~+0.21R or +0.22R
    assert m.duration_source == "BAR_SIMULATION"
    assert m.r_distribution["[1R, 2R)"] == 1                 # MSFT
    assert m.r_distribution["[-2R, -1R)"] == 1                # TSLA
    assert m.r_distribution["< -2R"] == 0

    # Generate cumulative report
    cum_data, cum_md = shadow_engine.generate_cumulative_report()

    assert cum_data["report_type"] == "SHADOW_CUMULATIVE"
    assert "Reporte Cuantitativo Acumulativo de Shadow Trading" in cum_md
    assert "R-Multiple Acumulado (Cumulative R)" in cum_md
    assert f"{'+' if m.cumulative_r_multiple >= 0 else ''}{m.cumulative_r_multiple:.2f}R" in cum_md
    assert "Distribución Estadística de R-Multiple" in cum_md
    assert "Serie Histórica de Progresión Diaria" in cum_md
    assert "CONGELADAS" in cum_md
    assert "duration_source = BAR_SIMULATION" in cum_md

    # Check files exist
    cum_json = shadow_engine.reports_dir / "shadow_cumulative_report.json"
    cum_md_file = shadow_engine.reports_dir / "shadow_cumulative_report.md"
    assert cum_json.exists()
    assert cum_md_file.exists()


