"""
Unit tests for PreTradeRiskGuard & GapRiskEngine.
"""

from __future__ import annotations
import datetime
from zoneinfo import ZoneInfo
import pytest

from core.market_session import MarketSession, SessionCapabilities
from services.risk_engine.pre_trade_guard import GapRiskEngine, PreTradeRiskGuard

ET = ZoneInfo("America/New_York")


class TestGapRiskEngine:

    def test_normal_gap_approved(self):
        res = GapRiskEngine.evaluate(
            symbol="AAPL",
            direction="buy",
            current_price=151.0,
            prev_close=150.0,
            natr_14=2.0,
            news_score=20.0,
        )
        assert res.status == "APPROVED"
        assert res.size_multiplier == 1.0

    def test_extreme_gap_without_catalyst_rejected(self):
        # 8% gap with 2% NATR = 4x NATR, news_score is 10.0 (no catalyst)
        res = GapRiskEngine.evaluate(
            symbol="NVDA",
            direction="buy",
            current_price=108.0,
            prev_close=100.0,
            natr_14=2.0,
            news_score=10.0,
        )
        assert res.status == "REJECT"
        assert res.size_multiplier == 0.0

    def test_extreme_gap_with_strong_catalyst_reduced(self):
        # 8% gap with news_score = 75.0 -> Allowed with 50% size
        res = GapRiskEngine.evaluate(
            symbol="NVDA",
            direction="buy",
            current_price=108.0,
            prev_close=100.0,
            natr_14=2.0,
            news_score=75.0,
        )
        assert res.status == "REDUCE"
        assert res.size_multiplier == 0.5


class TestPreTradeRiskGuard:

    def test_weekend_entry_blocked(self):
        guard = PreTradeRiskGuard()
        caps = SessionCapabilities(
            session=MarketSession.CLOSED,
            granular_session="weekend_analysis",
            can_analyze=True,
            can_generate_signal=True,
            can_enter=False,
            can_exit=False,
            can_manage_stop=False,
            can_cancel=False,
            can_place_limit=False,
            can_place_market=False,
            can_fetch_news=True,
            is_premarket=False,
            is_regular=False,
            is_afterhours=False,
            is_weekend=True,
            is_near_close=False,
            max_position_size_multiplier=0.0,
            recommended_cycle_seconds=1800.0,
        )
        res = guard.evaluate_order_proposal(
            symbol="AAPL",
            direction="buy",
            current_price=150.0,
            prev_close=150.0,
            capabilities=caps,
        )
        assert res.approved is False
        assert any("prohíbe nuevas entradas" in v for v in res.violations)

    def test_regular_session_approved(self):
        guard = PreTradeRiskGuard()
        caps = SessionCapabilities(
            session=MarketSession.REGULAR,
            granular_session="regular_full",
            can_analyze=True,
            can_generate_signal=True,
            can_enter=True,
            can_exit=True,
            can_manage_stop=True,
            can_cancel=True,
            can_place_limit=True,
            can_place_market=True,
            can_fetch_news=True,
            is_premarket=False,
            is_regular=True,
            is_afterhours=False,
            is_weekend=False,
            is_near_close=False,
            max_position_size_multiplier=1.0,
            recommended_cycle_seconds=60.0,
        )
        res = guard.evaluate_order_proposal(
            symbol="AAPL",
            direction="buy",
            current_price=150.0,
            prev_close=149.5,
            spread_pct=0.0005,
            capabilities=caps,
        )
        assert res.approved is True
        assert res.size_multiplier == 1.0
