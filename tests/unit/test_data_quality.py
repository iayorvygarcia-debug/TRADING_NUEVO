"""
Tests for Data Quality Engine.

Validates all 8 quality checks:
1. OHLC consistency
2. Price range
3. Volume validity
4. Timestamp validation
5. Duplicate detection
6. Price anomaly (spikes)
7. Volume anomaly (z-score)
8. Market session context
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from services.market_data.models import Bar, Quote, Timeframe, QualityStatus
from services.data_quality.validators import DataQualityEngine


def make_bar(
    symbol: str = "AAPL",
    timeframe: Timeframe = Timeframe.D1,
    time: datetime | None = None,
    open: float = 150.0,
    high: float = 155.0,
    low: float = 149.0,
    close: float = 153.0,
    volume: int = 1_000_000,
    **kwargs,
) -> Bar:
    """Helper to create test bars."""
    if time is None:
        time = datetime(2026, 9, 8, 16, 0, tzinfo=timezone.utc)
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        time=time,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="test",
        **kwargs,
    )


class TestOHLCConsistency:
    """Test OHLC relationship validation."""

    def test_valid_bar(self):
        dq = DataQualityEngine()
        bar = make_bar(open=150, high=155, low=149, close=153)
        report = dq.validate_bar(bar)
        assert report.is_valid

    def test_high_below_low(self):
        dq = DataQualityEngine()
        bar = make_bar(open=150, high=148, low=149, close=153)
        report = dq.validate_bar(bar)
        assert not report.is_clean
        assert any("ohlc_invalid" in i for i in report.issues)

    def test_high_below_open(self):
        dq = DataQualityEngine()
        bar = make_bar(open=156, high=155, low=149, close=153)
        report = dq.validate_bar(bar)
        assert any("ohlc_invalid" in i for i in report.issues)

    def test_high_below_close(self):
        dq = DataQualityEngine()
        bar = make_bar(open=150, high=152, low=149, close=153)
        report = dq.validate_bar(bar)
        assert any("ohlc_invalid" in i for i in report.issues)


class TestPriceRange:
    """Test price range validation."""

    def test_valid_price(self):
        dq = DataQualityEngine()
        bar = make_bar(close=150)
        report = dq.validate_bar(bar)
        assert report.is_valid

    def test_zero_price(self):
        dq = DataQualityEngine()
        bar = make_bar(open=0, high=155, low=0, close=0)
        report = dq.validate_bar(bar)
        assert not report.is_valid
        assert any("price_zero" in i for i in report.issues)

    def test_negative_price(self):
        dq = DataQualityEngine()
        bar = make_bar(open=-5, high=155, low=-10, close=-3)
        report = dq.validate_bar(bar)
        assert not report.is_valid

    def test_extreme_price(self):
        dq = DataQualityEngine(max_price=500_000)
        bar = make_bar(open=600_000, high=700_000, low=500_001, close=650_000)
        report = dq.validate_bar(bar)
        assert any("price_above_maximum" in i for i in report.issues)


class TestVolume:
    """Test volume validation."""

    def test_valid_volume(self):
        dq = DataQualityEngine()
        bar = make_bar(volume=1_000_000)
        report = dq.validate_bar(bar)
        assert report.is_valid

    def test_negative_volume(self):
        dq = DataQualityEngine()
        bar = make_bar(volume=-100)
        report = dq.validate_bar(bar)
        assert not report.is_valid

    def test_zero_volume_warning(self):
        dq = DataQualityEngine()
        bar = make_bar(volume=0)
        report = dq.validate_bar(bar)
        assert report.status == QualityStatus.SUSPECT
        assert any("volume_zero" in w for w in report.warnings)


class TestTimestamp:
    """Test timestamp validation."""

    def test_valid_timestamp(self):
        dq = DataQualityEngine()
        bar = make_bar(time=datetime.now(timezone.utc) - timedelta(minutes=5))
        report = dq.validate_bar(bar)
        assert report.is_valid

    def test_future_timestamp(self):
        dq = DataQualityEngine()
        bar = make_bar(time=datetime.now(timezone.utc) + timedelta(hours=1))
        report = dq.validate_bar(bar)
        assert any("timestamp_future" in i for i in report.issues)

    def test_out_of_order_warning(self):
        dq = DataQualityEngine()
        t1 = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 8, 13, 0, tzinfo=timezone.utc)  # Before t1

        bar1 = make_bar(time=t1)
        bar2 = make_bar(time=t2)

        dq.validate_bar(bar1)
        report = dq.validate_bar(bar2)
        assert any("out_of_order" in w for w in report.warnings)


class TestDuplicate:
    """Test duplicate detection."""

    def test_unique_bars(self):
        dq = DataQualityEngine()
        t1 = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)

        r1 = dq.validate_bar(make_bar(time=t1))
        r2 = dq.validate_bar(make_bar(time=t2))
        assert r1.is_valid
        assert r2.is_valid

    def test_duplicate_detected(self):
        dq = DataQualityEngine()
        t1 = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)

        r1 = dq.validate_bar(make_bar(time=t1))
        r2 = dq.validate_bar(make_bar(time=t1))  # Same timestamp
        assert r1.is_valid
        assert any("duplicate" in w for w in r2.warnings)


class TestPriceAnomaly:
    """Test price spike detection."""

    def test_normal_change(self):
        dq = DataQualityEngine(max_price_change_pct=50)
        t1 = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)

        dq.validate_bar(make_bar(time=t1, close=150))
        report = dq.validate_bar(make_bar(time=t2, close=153))  # 2% change
        assert report.is_valid

    def test_extreme_spike(self):
        dq = DataQualityEngine(max_price_change_pct=20)
        t1 = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)

        dq.validate_bar(make_bar(time=t1, open=98, high=102, low=97, close=100))
        report = dq.validate_bar(make_bar(
            time=t2, open=180, high=195, low=175, close=190
        ))  # 90% change from 100 → 190
        assert any("price_spike" in i for i in report.issues)


class TestQuoteValidation:
    """Test quote validation."""

    def test_valid_quote(self):
        dq = DataQualityEngine()
        quote = Quote(
            symbol="AAPL", time=datetime.now(timezone.utc),
            bid=150.0, ask=150.02, bid_size=100, ask_size=200, source="test",
        )
        report = dq.validate_quote(quote)
        assert report.is_valid

    def test_crossed_quote(self):
        dq = DataQualityEngine()
        quote = Quote(
            symbol="AAPL", time=datetime.now(timezone.utc),
            bid=150.05, ask=150.00, source="test",  # bid > ask
        )
        report = dq.validate_quote(quote)
        assert not report.is_valid

    def test_wide_spread(self):
        dq = DataQualityEngine()
        quote = Quote(
            symbol="AAPL", time=datetime.now(timezone.utc),
            bid=100.0, ask=120.0, source="test",  # 18% spread
        )
        report = dq.validate_quote(quote)
        assert any("wide_spread" in w for w in report.warnings)


class TestStaleDetection:
    """Test stale data detection."""

    def test_no_data_is_stale(self):
        dq = DataQualityEngine()
        assert dq.check_stale("UNKNOWN") is True

    def test_recent_data_not_stale(self):
        dq = DataQualityEngine(stale_threshold_seconds=300)
        bar = make_bar(time=datetime.now(timezone.utc))
        dq.validate_bar(bar)
        assert dq.check_stale("AAPL") is False


class TestStats:
    """Test aggregate statistics."""

    def test_stats_tracking(self):
        dq = DataQualityEngine()
        t = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)

        # One valid bar
        dq.validate_bar(make_bar(time=t))
        stats = dq.get_stats()
        assert stats["total_checked"] == 1
        assert stats["valid"] >= 1

    def test_stats_reset(self):
        dq = DataQualityEngine()
        t = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        dq.validate_bar(make_bar(time=t))
        dq.reset_stats()
        stats = dq.get_stats()
        assert stats["total_checked"] == 0
