"""
Tests for Market Data Models.

Validates Bar, Quote, Trade model properties and validation.
"""

from __future__ import annotations

from datetime import datetime, timezone

from services.market_data.models import Bar, Quote, Trade, MarketSnapshot, Timeframe, QualityStatus


class TestBar:
    """Test Bar model."""

    def test_bar_creation(self):
        bar = Bar(
            symbol="AAPL",
            timeframe=Timeframe.D1,
            time=datetime(2026, 9, 8, 16, 0, tzinfo=timezone.utc),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1_000_000,
            source="test",
        )
        assert bar.symbol == "AAPL"
        assert bar.close == 153.0

    def test_bar_is_bullish(self):
        bar = Bar(
            symbol="AAPL", timeframe=Timeframe.D1,
            time=datetime.now(timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0,
            volume=1_000_000,
        )
        assert bar.is_bullish is True

    def test_bar_is_bearish(self):
        bar = Bar(
            symbol="AAPL", timeframe=Timeframe.D1,
            time=datetime.now(timezone.utc),
            open=153.0, high=155.0, low=149.0, close=150.0,
            volume=1_000_000,
        )
        assert bar.is_bullish is False

    def test_bar_range(self):
        bar = Bar(
            symbol="AAPL", timeframe=Timeframe.D1,
            time=datetime.now(timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0,
            volume=1_000_000,
        )
        assert bar.range == 6.0  # 155 - 149

    def test_bar_body(self):
        bar = Bar(
            symbol="AAPL", timeframe=Timeframe.D1,
            time=datetime.now(timezone.utc),
            open=150.0, high=155.0, low=149.0, close=153.0,
            volume=1_000_000,
        )
        assert bar.body == 3.0  # |153 - 150|

    def test_bar_mid(self):
        bar = Bar(
            symbol="AAPL", timeframe=Timeframe.D1,
            time=datetime.now(timezone.utc),
            open=150.0, high=156.0, low=148.0, close=153.0,
            volume=1_000_000,
        )
        assert bar.mid == 152.0  # (156 + 148) / 2


class TestQuote:
    """Test Quote model."""

    def test_quote_spread(self):
        quote = Quote(
            symbol="AAPL",
            time=datetime.now(timezone.utc),
            bid=150.00,
            ask=150.05,
        )
        assert quote.spread == pytest.approx(0.05)

    def test_quote_spread_pct(self):
        quote = Quote(
            symbol="AAPL",
            time=datetime.now(timezone.utc),
            bid=100.0,
            ask=100.10,
        )
        assert quote.spread_pct == pytest.approx(0.001, rel=1e-3)


class TestTimeframe:
    """Test Timeframe enum."""

    def test_timeframe_values(self):
        assert Timeframe.M1.value == "1m"
        assert Timeframe.D1.value == "1d"
        assert Timeframe.H4.value == "4h"

    def test_timeframe_from_string(self):
        assert Timeframe("1d") == Timeframe.D1
        assert Timeframe("5m") == Timeframe.M5


import pytest
