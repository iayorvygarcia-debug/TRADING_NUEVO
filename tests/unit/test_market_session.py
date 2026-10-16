"""
Tests for Market Session Engine.

Validates:
- Correct session detection for all times of day
- Holiday detection
- Early close detection
- Weekend detection
- can_trade / is_regular_hours logic
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import pytest

from core.market_session import (
    MarketSession,
    get_current_session,
    can_trade,
    is_regular_hours,
    is_holiday,
    is_early_close,
    get_session_info,
)

ET = ZoneInfo("America/New_York")


class TestMarketSessions:
    """Test session detection at various times."""

    def test_premarket(self):
        """04:00-09:29 ET should be PREMARKET."""
        dt = datetime.datetime(2026, 9, 8, 7, 0, tzinfo=ET)  # Tuesday 7:00 AM
        assert get_current_session(dt) == MarketSession.PREMARKET

    def test_regular_open(self):
        """09:30 ET should be REGULAR."""
        dt = datetime.datetime(2026, 9, 8, 9, 30, tzinfo=ET)  # Tuesday 9:30 AM
        assert get_current_session(dt) == MarketSession.REGULAR

    def test_regular_mid(self):
        """12:00 ET should be REGULAR."""
        dt = datetime.datetime(2026, 9, 8, 12, 0, tzinfo=ET)
        assert get_current_session(dt) == MarketSession.REGULAR

    def test_regular_close(self):
        """15:59 ET should be REGULAR."""
        dt = datetime.datetime(2026, 9, 8, 15, 59, tzinfo=ET)
        assert get_current_session(dt) == MarketSession.REGULAR

    def test_afterhours(self):
        """16:00-19:59 ET should be AFTERHOURS."""
        dt = datetime.datetime(2026, 9, 8, 17, 0, tzinfo=ET)
        assert get_current_session(dt) == MarketSession.AFTERHOURS

    def test_closed_night(self):
        """22:00 ET should be CLOSED."""
        dt = datetime.datetime(2026, 9, 8, 22, 0, tzinfo=ET)
        assert get_current_session(dt) == MarketSession.CLOSED

    def test_closed_early_morning(self):
        """03:00 ET should be CLOSED."""
        dt = datetime.datetime(2026, 9, 8, 3, 0, tzinfo=ET)
        assert get_current_session(dt) == MarketSession.CLOSED

    def test_weekend_saturday(self):
        """Saturday should be CLOSED."""
        dt = datetime.datetime(2026, 9, 12, 12, 0, tzinfo=ET)  # Saturday
        assert get_current_session(dt) == MarketSession.CLOSED

    def test_weekend_sunday(self):
        """Sunday should be CLOSED."""
        dt = datetime.datetime(2026, 9, 13, 12, 0, tzinfo=ET)  # Sunday
        assert get_current_session(dt) == MarketSession.CLOSED


class TestHolidays:
    """Test holiday detection."""

    def test_christmas_2026(self):
        """Dec 25 2026 is Friday → market closed."""
        assert is_holiday(datetime.date(2026, 12, 25))

    def test_new_years_2026(self):
        """Jan 1 2026 is Thursday → market closed."""
        assert is_holiday(datetime.date(2026, 1, 1))

    def test_normal_day_not_holiday(self):
        """Regular Tuesday should not be a holiday."""
        assert not is_holiday(datetime.date(2026, 9, 8))

    def test_holiday_session(self):
        """During a holiday, session should be HOLIDAY."""
        dt = datetime.datetime(2026, 12, 25, 12, 0, tzinfo=ET)
        assert get_current_session(dt) == MarketSession.HOLIDAY


class TestEarlyClose:
    """Test early close detection."""

    def test_black_friday_2026(self):
        """Day after Thanksgiving 2026 should be early close."""
        # Thanksgiving 2026 = Nov 26 (4th Thursday)
        assert is_early_close(datetime.date(2026, 11, 27))

    def test_early_close_session_before_13(self):
        """On early close day, 10:00 should be EARLY_CLOSE_REGULAR."""
        dt = datetime.datetime(2026, 11, 27, 10, 0, tzinfo=ET)
        session = get_current_session(dt)
        assert session == MarketSession.EARLY_CLOSE_REGULAR

    def test_early_close_session_after_13(self):
        """On early close day, 14:00 should be CLOSED."""
        dt = datetime.datetime(2026, 11, 27, 14, 0, tzinfo=ET)
        session = get_current_session(dt)
        assert session == MarketSession.CLOSED


class TestCanTrade:
    """Test trading availability."""

    def test_can_trade_regular(self):
        assert can_trade(MarketSession.REGULAR) is True

    def test_can_trade_premarket(self):
        assert can_trade(MarketSession.PREMARKET) is True

    def test_can_trade_afterhours(self):
        assert can_trade(MarketSession.AFTERHOURS) is True

    def test_cannot_trade_closed(self):
        assert can_trade(MarketSession.CLOSED) is False

    def test_cannot_trade_holiday(self):
        assert can_trade(MarketSession.HOLIDAY) is False


class TestIsRegularHours:
    """Test regular hours detection."""

    def test_regular_is_regular(self):
        assert is_regular_hours(MarketSession.REGULAR) is True

    def test_premarket_not_regular(self):
        assert is_regular_hours(MarketSession.PREMARKET) is False

    def test_afterhours_not_regular(self):
        assert is_regular_hours(MarketSession.AFTERHOURS) is False


class TestSessionInfo:
    """Test session info output."""

    def test_session_info_structure(self):
        dt = datetime.datetime(2026, 9, 8, 12, 0, tzinfo=ET)
        info = get_session_info(dt)

        assert "session" in info
        assert "can_trade" in info
        assert "is_regular" in info
        assert "timestamp" in info
        assert "timezone" in info
        assert "weekday" in info
        assert info["session"] == "regular"
        assert info["can_trade"] is True
        assert info["is_regular"] is True
        assert "capabilities" in info
        assert info["capabilities"]["can_enter"] is True


class TestSessionCapabilities:
    """Tests for granular decoupled SessionCapabilities."""

    def test_weekend_capabilities(self):
        """Weekend should allow analysis, but forbid entering, exiting, managing stops."""
        from core.market_session import get_session_capabilities
        sat = datetime.datetime(2026, 9, 12, 14, 0, tzinfo=ET)
        caps = get_session_capabilities(sat)

        assert caps.is_weekend is True
        assert caps.can_analyze is True
        assert caps.can_generate_signal is True
        assert caps.can_enter is False
        assert caps.can_exit is False
        assert caps.can_manage_stop is False
        assert caps.can_cancel is False
        assert caps.granular_session == "weekend_analysis"
        assert caps.recommended_cycle_seconds == 1800.0

    def test_premarket_early_capabilities(self):
        """04:00-07:00 ET: Illiquid, can exit (SL/emergencies) and cancel, but strictly no new entries."""
        from core.market_session import get_session_capabilities
        dt = datetime.datetime(2026, 9, 8, 5, 30, tzinfo=ET)
        caps = get_session_capabilities(dt)

        assert caps.granular_session == "premarket_early"
        assert caps.can_analyze is True
        assert caps.can_enter is False
        assert caps.can_exit is True
        assert caps.can_manage_stop is True
        assert caps.can_cancel is True
        assert caps.can_place_market is False

    def test_premarket_active_capabilities(self):
        """07:00-09:25 ET: Active premarket, limit orders allowed."""
        from core.market_session import get_session_capabilities
        dt = datetime.datetime(2026, 9, 8, 8, 30, tzinfo=ET)
        caps = get_session_capabilities(dt)

        assert caps.granular_session == "premarket_active"
        assert caps.can_enter is True
        assert caps.can_place_limit is True
        assert caps.can_place_market is False

    def test_near_close_capabilities(self):
        """15:40-16:00 ET: Risk lockdown, no new entries, exits and stop management permitted."""
        from core.market_session import get_session_capabilities
        dt = datetime.datetime(2026, 9, 8, 15, 45, tzinfo=ET)
        caps = get_session_capabilities(dt)

        assert caps.granular_session == "near_close"
        assert caps.is_near_close is True
        assert caps.can_enter is False
        assert caps.can_exit is True
        assert caps.max_position_size_multiplier == 0.0


class TestDynamicSpread:
    """Tests for dynamic spread model."""

    def test_dynamic_spread_mega_cap_vs_mid_cap(self):
        from core.market_session import get_dynamic_max_spread
        spread_aapl = get_dynamic_max_spread("AAPL", MarketSession.REGULAR)
        spread_mid = get_dynamic_max_spread("XYZ", MarketSession.REGULAR)

        # Mega-cap spread requirement should be stricter than mid-cap
        assert spread_aapl <= spread_mid

    def test_dynamic_spread_extended_hours_expansion(self):
        from core.market_session import get_dynamic_max_spread
        spread_reg = get_dynamic_max_spread("NVDA", MarketSession.REGULAR)
        spread_ext = get_dynamic_max_spread("NVDA", MarketSession.PREMARKET)

        # Extended hours allows wider spread
        assert spread_ext > spread_reg

