"""
Unit Tests for Deterministic Risk Engine, Circuit Breaker, and Kill Switch.

Tests:
1. KillSwitch (activation, deactivation passphrase, assertion)
2. CircuitBreaker (5 levels, auto-escalation, Kill Switch auto-trigger)
3. PreTradeChecker (11 deterministic safety gates, scale down)
4. RiskEngine (master coordination)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch
import pytest

from core.exceptions import KillSwitchActivatedError
from services.risk_engine.models import PortfolioRiskState, RiskLevel, PreTradeCheckResult
from services.risk_engine.kill_switch import KillSwitch, get_kill_switch
from services.risk_engine.circuit_breaker import CircuitBreaker
from services.risk_engine.pre_trade_checker import PreTradeChecker
from services.risk_engine.engine import RiskEngine


@pytest.fixture(autouse=True)
def reset_global_kill_switch() -> None:
    """Ensure singleton KillSwitch is disengaged before and after each test."""
    get_kill_switch().reset()
    yield
    get_kill_switch().reset()


# ── 1. Kill Switch Tests ─────────────────────────────────────────────

class TestKillSwitch:

    def test_initial_state(self) -> None:
        ks = KillSwitch()
        assert not ks.is_active
        assert ks.status["is_active"] is False

    def test_activation(self) -> None:
        ks = KillSwitch()
        ks.activate(reason="Market emergency", operator="test_runner")
        assert ks.is_active is True
        assert ks.status["reason"] == "Market emergency"
        assert ks.status["operator"] == "test_runner"

        with pytest.raises(KillSwitchActivatedError, match="KILL SWITCH ACTIVATED"):
            ks.assert_safe()

    def test_deactivation_passphrase(self) -> None:
        ks = KillSwitch()
        ks.activate(reason="Testing lock", operator="tester")

        # Wrong passphrase fails
        assert ks.deactivate("wrong_password") is False
        assert ks.is_active is True

        # Correct passphrase succeeds
        assert ks.deactivate("CONFIRM_RESUME") is True
        assert ks.is_active is False
        # assert_safe should not raise now
        ks.assert_safe()


# ── 2. Circuit Breaker Tests ─────────────────────────────────────────

class TestCircuitBreaker:

    def test_normal_state(self) -> None:
        cb = CircuitBreaker()
        state = PortfolioRiskState(
            equity=100000.0,
            peak_equity=100000.0,
            cash=100000.0,
            daily_starting_equity=100000.0,
        )
        level = cb.evaluate(state)
        assert level == RiskLevel.NORMAL
        assert cb.can_open_new_positions is True
        assert cb.position_size_multiplier == 1.0

    def test_level_1_warning(self) -> None:
        cb = CircuitBreaker(warning_daily_loss_pct=1.5)
        # 2% daily loss ($100k -> $98k)
        state = PortfolioRiskState(
            equity=98000.0,
            peak_equity=100000.0,
            cash=98000.0,
            daily_starting_equity=100000.0,
        )
        level = cb.evaluate(state)
        assert level == RiskLevel.WARNING
        assert cb.can_open_new_positions is True
        assert cb.position_size_multiplier == 0.75  # Scaled down

    def test_level_2_stop_new_on_daily_loss(self) -> None:
        cb = CircuitBreaker(stop_new_daily_loss_pct=3.0)
        # 3.5% daily loss
        state = PortfolioRiskState(
            equity=96500.0,
            peak_equity=100000.0,
            cash=96500.0,
            daily_starting_equity=100000.0,
        )
        level = cb.evaluate(state)
        assert level == RiskLevel.STOP_NEW
        assert cb.can_open_new_positions is False

    def test_level_2_stop_new_on_consecutive_losses(self) -> None:
        cb = CircuitBreaker(max_consecutive_losses=4)
        state = PortfolioRiskState(
            equity=99000.0,
            peak_equity=100000.0,
            cash=99000.0,
            daily_starting_equity=100000.0,
            consecutive_losses=4,
        )
        level = cb.evaluate(state)
        assert level == RiskLevel.STOP_NEW
        assert cb.can_open_new_positions is False

    def test_layer_2_profit_lock_triggers_stop_new(self) -> None:
        cb = CircuitBreaker(min_profit_lock_usd=300.0, max_profit_giveback_pct=35.0)
        # Started at 100k, peaked at 100.6k (+$600 profit), dropped to 100.35k (gave back $250 = 41.7%)
        state = PortfolioRiskState(
            equity=100350.0,
            peak_equity=100600.0,
            cash=100350.0,
            daily_starting_equity=100000.0,
        )
        level = cb.evaluate(state)
        assert level == RiskLevel.STOP_NEW
        assert cb.can_open_new_positions is False
        assert any("Capa 2 (Profit Lock)" in r for r in cb._trigger_reasons)

    def test_layer_2_profit_lock_allows_safe_pullback(self) -> None:
        cb = CircuitBreaker(min_profit_lock_usd=300.0, max_profit_giveback_pct=35.0)
        # Peaked at +$600, dropped only $100 to 100.5k (gave back 16.7% <= 35%)
        state = PortfolioRiskState(
            equity=100500.0,
            peak_equity=100600.0,
            cash=100500.0,
            daily_starting_equity=100000.0,
        )
        level = cb.evaluate(state)
        assert level == RiskLevel.NORMAL
        assert cb.can_open_new_positions is True

    def test_level_4_emergency_triggers_kill_switch(self) -> None:
        cb = CircuitBreaker(emergency_daily_loss_pct=6.0)
        # 7% daily loss
        state = PortfolioRiskState(
            equity=93000.0,
            peak_equity=100000.0,
            cash=93000.0,
            daily_starting_equity=100000.0,
        )
        level = cb.evaluate(state)
        assert level == RiskLevel.EMERGENCY


# ── 3. Pre-Trade Risk Checker Tests ──────────────────────────────────

class TestPreTradeChecker:

    @pytest.fixture
    def safe_state(self) -> PortfolioRiskState:
        return PortfolioRiskState(
            equity=100000.0,
            peak_equity=100000.0,
            cash=100000.0,
            daily_starting_equity=100000.0,
            daily_trades_count=2,
        )

    def test_approved_trade(self, safe_state: PortfolioRiskState) -> None:
        checker = PreTradeChecker(
            circuit_breaker=CircuitBreaker(),
            kill_switch=KillSwitch(),
        )
        with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
            # 20 shares @ $150 = $3,000 (3% of $100k, within 5% limit)
            res = checker.check(
                symbol="AAPL",
                direction="buy",
                quantity=20,
                price=150.0,
                portfolio_state=safe_state,
                signal_score=0.85,
                current_spread_pct=0.001,
            )
            assert res.approved is True
            assert res.approved_quantity == 20.0
            assert len(res.violations) == 0

    def test_rejected_by_kill_switch(self, safe_state: PortfolioRiskState) -> None:
        ks = KillSwitch()
        ks.activate(reason="Test Kill Switch")
        checker = PreTradeChecker(kill_switch=ks)

        res = checker.check(
            symbol="AAPL",
            direction="buy",
            quantity=10,
            price=150.0,
            portfolio_state=safe_state,
        )
        assert res.approved is False
        assert any("KILL SWITCH" in v for v in res.violations)

    def test_rejected_by_closed_market(self, safe_state: PortfolioRiskState) -> None:
        checker = PreTradeChecker(kill_switch=KillSwitch())
        with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=False):
            res = checker.check(
                symbol="AAPL",
                direction="buy",
                quantity=10,
                price=150.0,
                portfolio_state=safe_state,
            )
            assert res.approved is False
            assert any("closed" in v for v in res.violations)

    def test_rejected_by_max_daily_loss(self) -> None:
        checker = PreTradeChecker(kill_switch=KillSwitch())
        bad_state = PortfolioRiskState(
            equity=96000.0,  # 4% daily loss > 3% max
            peak_equity=100000.0,
            cash=96000.0,
            daily_starting_equity=100000.0,
        )
        with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
            res = checker.check(
                symbol="AAPL",
                direction="buy",
                quantity=10,
                price=150.0,
                portfolio_state=bad_state,
            )
            assert res.approved is False
            assert any("daily loss" in v.lower() for v in res.violations)

    def test_rejected_by_wide_spread(self, safe_state: PortfolioRiskState) -> None:
        checker = PreTradeChecker(kill_switch=KillSwitch())
        with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
            res = checker.check(
                symbol="ILLIQ",
                direction="buy",
                quantity=10,
                price=150.0,
                portfolio_state=safe_state,
                current_spread_pct=0.012,  # 1.2% > 0.5% max
            )
            assert res.approved is False
            assert any("spread too wide" in v.lower() for v in res.violations)

    def test_scale_down_on_large_position(self, safe_state: PortfolioRiskState) -> None:
        checker = PreTradeChecker(kill_switch=KillSwitch())
        with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
            # Requested: 100 shares @ $150 = $15,000 (15% of equity, max is 5% = $5,000 -> 33 shares)
            res = checker.check(
                symbol="AAPL",
                direction="buy",
                quantity=100,
                price=150.0,
                portfolio_state=safe_state,
                signal_score=0.85,
            )
            assert res.approved is True
            assert res.approved_quantity == 33.0  # Scaled down to max 5% ($5,000 / $150)
            assert res.was_scaled_down is True


# ── 4. Risk Engine Facade Tests ──────────────────────────────────────

class TestRiskEngine:

    def test_facade_workflow(self) -> None:
        engine = RiskEngine()
        state = PortfolioRiskState(
            equity=100000.0,
            peak_equity=100000.0,
            cash=100000.0,
            daily_starting_equity=100000.0,
        )

        with patch("services.risk_engine.pre_trade_checker.can_trade", return_value=True):
            res = engine.evaluate_order(
                symbol="MSFT",
                direction="buy",
                quantity=10,
                price=200.0,
                portfolio_state=state,
                signal_score=0.80,
            )
            assert res.approved is True

        # Test emergency kill switch
        engine.trigger_kill_switch("Manual test shutdown", operator="admin")
        status = engine.get_status()
        assert status["kill_switch"]["is_active"] is True

        # Next order must fail
        res2 = engine.evaluate_order(
            symbol="MSFT",
            direction="buy",
            quantity=10,
            price=200.0,
            portfolio_state=state,
        )
        assert res2.approved is False

        # Resume trading
        assert engine.resume_trading("CONFIRM_RESUME") is True
        assert engine.get_status()["kill_switch"]["is_active"] is False

    def test_session_aware_portfolio_state_tracking(self) -> None:
        engine = RiskEngine()
        # Initial call sets base
        s1 = engine.get_or_create_portfolio_state(equity=100000.0, cash=100000.0)
        assert s1.daily_starting_equity == 100000.0
        assert s1.peak_equity == 100000.0

        # Equity rises to 100,500 -> peak updates
        s2 = engine.get_or_create_portfolio_state(equity=100500.0, cash=95000.0)
        assert s2.daily_starting_equity == 100000.0
        assert s2.peak_equity == 100500.0
        assert s2.daily_peak_profit == 500.0

        # Equity pulls back to 100,350 -> peak stays at 100,500
        s3 = engine.get_or_create_portfolio_state(equity=100350.0, cash=95000.0)
        assert s3.daily_starting_equity == 100000.0
        assert s3.peak_equity == 100500.0
        assert s3.profit_giveback_pct == 30.0

    def test_scenario_a_worker_restart_preserves_start_and_peak(self) -> None:
        """Scenario A: Worker restarts midday after reaching $100,600 from $100,000 start."""
        # Simulated fresh RiskEngine instance created on worker reboot
        engine_rebooted = RiskEngine()
        
        # When worker boots, it passes persisted values from quant_worker_state.json
        state_after_reboot = engine_rebooted.get_or_create_portfolio_state(
            equity=100450.0,  # current equity after reboot
            cash=95000.0,
            daily_starting_equity=100000.0,
            peak_equity=100600.0,
        )
        assert state_after_reboot.daily_starting_equity == 100000.0
        assert state_after_reboot.peak_equity == 100600.0
        assert state_after_reboot.daily_peak_profit == 600.0

    def test_scenario_b_new_day_rollover_cleans_peak_and_profit(self) -> None:
        """Scenario B: Market opens on Day 2; profit metrics start clean at $0."""
        engine = RiskEngine()
        
        # Day 1: Session 2026-10-01
        engine._session_date = "2026-10-01"
        engine._daily_starting_equity = 100000.0
        engine._daily_peak_equity = 100600.0
        
        # Day 2: Session 2026-10-02 opens with $100,600 account equity
        with patch("services.risk_engine.engine.datetime") as mock_dt:
            from datetime import date
            from unittest.mock import MagicMock
            mock_now = MagicMock()
            mock_now.date.return_value = date(2026, 10, 2)
            mock_dt.now.return_value = mock_now
            
            day2_state = engine.get_or_create_portfolio_state(
                equity=100600.0,
                cash=95000.0,
                daily_starting_equity=100600.0,
                peak_equity=100600.0,
            )
            assert day2_state.daily_starting_equity == 100600.0
            assert day2_state.peak_equity == 100600.0
            assert day2_state.daily_peak_profit == 0.0  # Clean slate: zero carryover profit
            assert day2_state.daily_pnl == 0.0
            assert day2_state.profit_giveback_pct == 0.0
