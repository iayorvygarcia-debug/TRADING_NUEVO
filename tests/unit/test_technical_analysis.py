"""
Unit Tests for Technical Analysis Service.

Tests:
1. TechnicalIndicators (Trend, Momentum, Volatility, Volume, S&R)
2. CandlePatternDetector (Single-bar, Two-bar, Three-bar, FVG)
3. MultiTimeframeAggregator (Resampling, Alignment, Pullbacks)
4. FeatureEngine (FeatureVector, to_ml_features)
5. FeatureStore (Caching, snapshots, history, dataframe export)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest

from services.technical_analysis.indicators import (
    TechnicalIndicators,
    calculate_all_indicators,
)
from services.technical_analysis.patterns import (
    CandlePatternDetector,
    PatternSentiment,
)
from services.technical_analysis.multi_timeframe import (
    MultiTimeframeAggregator,
    TrendDirection,
)
from services.technical_analysis.feature_engine import (
    FeatureEngine,
    FeatureVector,
)
from services.technical_analysis.feature_store import (
    FeatureStore,
    get_feature_store,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 100 bars of realistic synthetic OHLCV data."""
    np.random.seed(42)
    n = 100
    start_time = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    times = [start_time + timedelta(minutes=i) for i in range(n)]

    # Random walk for close prices
    returns = np.random.normal(loc=0.0005, scale=0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))

    opens = prices + np.random.uniform(-0.5, 0.5, size=n)
    highs = np.maximum(opens, prices) + np.random.uniform(0.1, 1.0, size=n)
    lows = np.minimum(opens, prices) - np.random.uniform(0.1, 1.0, size=n)
    closes = prices
    volumes = np.random.randint(10000, 100000, size=n).astype(float)

    df = pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    df.set_index("time", inplace=True)
    return df


# ── 1. Technical Indicators Tests ────────────────────────────────────

class TestTechnicalIndicators:

    def test_moving_averages(self, sample_ohlcv: pd.DataFrame) -> None:
        c = sample_ohlcv["close"]
        sma = TechnicalIndicators.sma(c, 20)
        ema = TechnicalIndicators.ema(c, 20)
        wma = TechnicalIndicators.wma(c, 20)
        hma = TechnicalIndicators.hma(c, 20)

        assert len(sma) == len(c)
        assert np.isnan(sma.iloc[0])  # Warmup
        assert not np.isnan(sma.iloc[-1])
        assert not np.isnan(ema.iloc[-1])
        assert not np.isnan(wma.iloc[-1])
        assert not np.isnan(hma.iloc[-1])

    def test_rsi(self, sample_ohlcv: pd.DataFrame) -> None:
        c = sample_ohlcv["close"]
        rsi = TechnicalIndicators.rsi(c, 14)
        valid = rsi.dropna()
        assert len(valid) > 0
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_macd(self, sample_ohlcv: pd.DataFrame) -> None:
        c = sample_ohlcv["close"]
        macd_l, signal_l, hist = TechnicalIndicators.macd(c, 12, 26, 9)
        assert not np.isnan(macd_l.iloc[-1])
        assert not np.isnan(signal_l.iloc[-1])
        assert not np.isnan(hist.iloc[-1])
        # Histogram should equal macd - signal
        assert pytest.approx(hist.iloc[-1], abs=1e-6) == (macd_l.iloc[-1] - signal_l.iloc[-1])

    def test_bollinger_bands(self, sample_ohlcv: pd.DataFrame) -> None:
        c = sample_ohlcv["close"]
        upper, mid, lower, pct_b, bw = TechnicalIndicators.bollinger_bands(c, 20, 2.0)
        assert (upper.dropna() >= mid.dropna()).all()
        assert (mid.dropna() >= lower.dropna()).all()
        assert (bw.dropna() >= 0.0).all()

    def test_atr_and_natr(self, sample_ohlcv: pd.DataFrame) -> None:
        h, l, c = sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        atr = TechnicalIndicators.atr(h, l, c, 14)
        natr = TechnicalIndicators.natr(h, l, c, 14)
        assert (atr.dropna() > 0.0).all()
        assert (natr.dropna() > 0.0).all()

    def test_supertrend(self, sample_ohlcv: pd.DataFrame) -> None:
        h, l, c = sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        st_val, st_dir = TechnicalIndicators.supertrend(h, l, c, 10, 3.0)
        assert st_dir.iloc[-1] in (1, -1)
        assert not np.isnan(st_val.iloc[-1])

    def test_adx(self, sample_ohlcv: pd.DataFrame) -> None:
        h, l, c = sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        adx, p_di, m_di = TechnicalIndicators.adx(h, l, c, 14)
        val_adx = adx.dropna()
        assert len(val_adx) > 0
        assert (val_adx >= 0.0).all()
        assert (val_adx <= 100.0).all()

    def test_volume_indicators(self, sample_ohlcv: pd.DataFrame) -> None:
        h, l, c, v = sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], sample_ohlcv["volume"]
        vwap = TechnicalIndicators.vwap(h, l, c, v)
        obv = TechnicalIndicators.obv(c, v)
        cmf = TechnicalIndicators.cmf(h, l, c, v, 20)
        mfi = TechnicalIndicators.mfi(h, l, c, v, 14)
        rvol = TechnicalIndicators.rvol(v, 20)

        assert not np.isnan(vwap.iloc[-1])
        assert not np.isnan(obv.iloc[-1])
        assert -1.0 <= cmf.iloc[-1] <= 1.0
        assert 0.0 <= mfi.iloc[-1] <= 100.0
        assert rvol.iloc[-1] > 0.0

    def test_pivot_points(self) -> None:
        pivots = TechnicalIndicators.pivot_points(high=150.0, low=140.0, close=145.0)
        assert pivots["pivot"] == 145.0
        assert pivots["r1"] > pivots["pivot"]
        assert pivots["s1"] < pivots["pivot"]
        assert pivots["cam_r4"] > pivots["cam_r3"]

    def test_calculate_all_indicators(self, sample_ohlcv: pd.DataFrame) -> None:
        res = calculate_all_indicators(sample_ohlcv)
        expected_cols = [
            "sma_20", "ema_9", "rsi_14", "macd_hist", "stoch_k",
            "bb_upper", "supertrend", "adx", "vwap", "cmf_20", "rvol_20"
        ]
        for col in expected_cols:
            assert col in res.columns


# ── 2. Candlestick Patterns Tests ────────────────────────────────────

class TestCandlePatterns:

    def test_detect_doji(self) -> None:
        bar = pd.Series({"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.2})
        patterns = CandlePatternDetector.detect_single_bar(bar, bar_idx=5)
        names = [p.pattern_name for p in patterns]
        assert "doji" in names

    def test_detect_hammer(self) -> None:
        # Small body at top, very long lower shadow
        bar = pd.Series({"open": 108.0, "high": 110.0, "low": 90.0, "close": 109.5})
        patterns = CandlePatternDetector.detect_single_bar(bar, bar_idx=10)
        names = [p.pattern_name for p in patterns]
        assert "hammer" in names or "bullish_pinbar" in names

    def test_detect_bullish_engulfing(self) -> None:
        prev = pd.Series({"open": 105.0, "high": 106.0, "low": 99.0, "close": 100.0})   # Bearish
        curr = pd.Series({"open": 99.0, "high": 110.0, "low": 98.0, "close": 107.0})    # Bullish engulfing
        patterns = CandlePatternDetector.detect_two_bar(prev, curr, bar_idx=2)
        names = [p.pattern_name for p in patterns]
        assert "bullish_engulfing" in names

    def test_detect_bearish_engulfing(self) -> None:
        prev = pd.Series({"open": 100.0, "high": 106.0, "low": 99.0, "close": 105.0})   # Bullish
        curr = pd.Series({"open": 106.0, "high": 107.0, "low": 98.0, "close": 98.5})    # Bearish engulfing
        patterns = CandlePatternDetector.detect_two_bar(prev, curr, bar_idx=2)
        names = [p.pattern_name for p in patterns]
        assert "bearish_engulfing" in names

    def test_detect_morning_star(self) -> None:
        p2 = pd.Series({"open": 110.0, "high": 111.0, "low": 99.0, "close": 100.0})   # Big bear
        p1 = pd.Series({"open": 98.0, "high": 99.0, "low": 96.0, "close": 97.5})       # Small star
        c = pd.Series({"open": 98.0, "high": 108.0, "low": 97.0, "close": 107.0})      # Big bull past midpoint
        patterns = CandlePatternDetector.detect_three_bar(p2, p1, c, bar_idx=3)
        names = [p.pattern_name for p in patterns]
        assert "morning_star" in names

    def test_detect_fvg(self) -> None:
        # Bullish FVG: Bar 1 High < Bar 3 Low
        b1 = pd.Series({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0})
        b2 = pd.Series({"open": 101.0, "high": 110.0, "low": 101.0, "close": 109.0})  # Huge impulse
        b3 = pd.Series({"open": 109.0, "high": 112.0, "low": 105.0, "close": 111.0})  # Low (105) > b1 High (102)

        fvg = CandlePatternDetector.detect_fvg(b1, b2, b3, bar_idx=5)
        assert fvg is not None
        assert fvg.pattern_name == "bullish_fvg"
        assert fvg.sentiment == PatternSentiment.BULLISH
        assert fvg.metadata["gap_size"] == 3.0


# ── 3. Multi-Timeframe Tests ─────────────────────────────────────────

class TestMultiTimeframe:

    def test_resample_1m_to_5m(self, sample_ohlcv: pd.DataFrame) -> None:
        df_5m = MultiTimeframeAggregator.resample(sample_ohlcv, "5m")
        assert len(df_5m) == len(sample_ohlcv) // 5
        assert set(df_5m.columns) == {"open", "high", "low", "close", "volume"}

    def test_trend_evaluation(self, sample_ohlcv: pd.DataFrame) -> None:
        direction, details = MultiTimeframeAggregator.evaluate_trend(sample_ohlcv)
        assert direction in (TrendDirection.BULLISH, TrendDirection.BEARISH, TrendDirection.NEUTRAL)
        assert "price" in details
        assert "ema_21" in details

    def test_analyze_alignment(self, sample_ohlcv: pd.DataFrame) -> None:
        alignment = MultiTimeframeAggregator.analyze_alignment(
            macro_df=sample_ohlcv,
            inter_df=sample_ohlcv,
            exec_df=sample_ohlcv,
            macro_tf="1d",
            inter_tf="1h",
            exec_tf="15m",
        )
        assert -1.0 <= alignment.alignment_score <= 1.0
        assert isinstance(alignment.is_aligned, bool)
        assert isinstance(alignment.is_pullback, bool)


# ── 4. Feature Engine Tests ──────────────────────────────────────────

class TestFeatureEngine:

    def test_compute_features(self, sample_ohlcv: pd.DataFrame) -> None:
        fv = FeatureEngine.compute_features(
            sample_ohlcv,
            symbol="AAPL",
            timeframe="1d",
        )
        assert isinstance(fv, FeatureVector)
        assert fv.symbol == "AAPL"
        assert fv.timeframe == "1d"
        assert fv.close_price > 0

        # Check dictionary conversion
        d = fv.to_dict()
        assert d["symbol"] == "AAPL"
        assert "rsi_14" in d
        assert "vwap_dist_pct" in d

        # Check ML feature array
        ml_vec = fv.to_ml_features()
        assert isinstance(ml_vec, np.ndarray)
        assert ml_vec.dtype == np.float32
        assert len(ml_vec) == len(FeatureVector.feature_names())
        assert not np.isnan(ml_vec).any()
        assert not np.isinf(ml_vec).any()

    def test_insufficient_bars_raises(self, sample_ohlcv: pd.DataFrame) -> None:
        small_df = sample_ohlcv.iloc[:10]
        with pytest.raises(ValueError, match="requires at least"):
            FeatureEngine.compute_features(small_df, symbol="AAPL")


# ── 5. Feature Store Tests ───────────────────────────────────────────

class TestFeatureStore:

    def test_save_and_get_latest(self, sample_ohlcv: pd.DataFrame) -> None:
        store = FeatureStore(max_history_per_key=10)
        fv = FeatureEngine.compute_features(sample_ohlcv, symbol="MSFT", timeframe="1h")

        store.save(fv)
        latest = store.get_latest("MSFT", "1h")
        assert latest is not None
        assert latest.symbol == "MSFT"
        assert latest.close_price == fv.close_price

    def test_universe_snapshot(self, sample_ohlcv: pd.DataFrame) -> None:
        store = FeatureStore()
        fv1 = FeatureEngine.compute_features(sample_ohlcv, symbol="AAPL", timeframe="1d")
        fv2 = FeatureEngine.compute_features(sample_ohlcv, symbol="NVDA", timeframe="1d")

        store.save(fv1)
        store.save(fv2)

        snapshot = store.get_universe_snapshot(["AAPL", "NVDA", "TSLA"], timeframe="1d")
        assert "AAPL" in snapshot
        assert "NVDA" in snapshot
        assert "TSLA" not in snapshot  # TSLA wasn't saved

    def test_to_dataframe(self, sample_ohlcv: pd.DataFrame) -> None:
        store = FeatureStore()
        fv = FeatureEngine.compute_features(sample_ohlcv, symbol="GOOGL", timeframe="1d")
        store.save(fv)

        df = store.to_dataframe("GOOGL", "1d")
        assert not df.empty
        assert "close_price" in df.columns
        assert "rsi_14" in df.columns
