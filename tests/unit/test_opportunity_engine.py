"""
Unit Tests for Opportunity Engine & Universe Scanner.

Tests:
1. UniverseManager (predefined universes, sector resolution)
2. OpportunityScreener (multi-stage filtering)
3. OpportunityScorer (4-factor composite scoring)
4. UniverseScanner (batch scanning, ranking, executive reporting)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest

from services.opportunity_engine.universe import Sector, UniverseManager
from services.opportunity_engine.screener import OpportunityScreener, ScreeningResult
from services.opportunity_engine.scorer import OpportunityScorer, ScoredOpportunity
from services.opportunity_engine.ranker import UniverseScanner
from services.technical_analysis.feature_engine import FeatureEngine, FeatureVector


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def active_feature_vector() -> FeatureVector:
    """A high-conviction bullish candidate feature vector."""
    return FeatureVector(
        symbol="NVDA",
        timestamp=datetime.now(timezone.utc),
        timeframe="1d",
        close_price=120.0,
        return_1=2.5,
        return_5=5.0,
        return_20=12.0,
        high_low_ratio=0.03,
        close_position_in_bar=0.85,
        ema_9_dist_pct=3.0,
        ema_21_dist_pct=5.5,
        ema_50_dist_pct=10.0,
        sma_200_dist_pct=25.0,
        ema_9_21_spread=1.8,
        supertrend_dir=1,
        adx_14=32.0,
        di_spread=12.0,
        rsi_14=62.0,
        macd_hist_norm=0.15,
        stoch_k=75.0,
        stoch_d=70.0,
        stoch_rsi_k=65.0,
        roc_10=8.0,
        cci_20=110.0,
        williams_r=-20.0,
        natr_14=2.8,
        bb_percent=0.78,
        bb_bandwidth=0.08,
        is_bb_squeeze=False,
        hist_vol_20=0.35,
        rvol_20=2.2,             # High volume surge
        cmf_20=0.18,             # Accumulation
        mfi_14=68.0,
        vwap_dist_pct=0.8,
        bullish_pattern_count=1,
        bearish_pattern_count=0,
        strongest_pattern="bullish_engulfing",
        pattern_sentiment_score=0.85,
        has_fvg=True,
        is_pinbar=False,
        is_inside_bar=False,
        mtf_alignment_score=0.8,
        mtf_is_pullback=False,
    )


# ── 1. Universe Manager Tests ────────────────────────────────────────

class TestUniverseManager:

    def test_predefined_universes(self) -> None:
        sp500 = UniverseManager.get_universe("sp500")
        assert "AAPL" in sp500
        assert "MSFT" in sp500
        assert len(sp500) == 50

        nasdaq = UniverseManager.get_universe("nasdaq")
        assert "NVDA" in nasdaq

        etfs = UniverseManager.get_universe("etfs")
        assert "SPY" in etfs
        assert "QQQ" in etfs

        all_syms = UniverseManager.get_universe("all")
        assert len(all_syms) > 50

    def test_unknown_universe_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown universe"):
            UniverseManager.get_universe("crypto_defi")

    def test_sector_resolution(self) -> None:
        assert UniverseManager.get_sector("AAPL") == Sector.TECHNOLOGY
        assert UniverseManager.get_sector("JPM") == Sector.FINANCIALS
        assert UniverseManager.get_sector("SPY") == Sector.ETF
        assert UniverseManager.get_sector("RANDOM_TICKER") == Sector.OTHER


# ── 2. Opportunity Screener Tests ────────────────────────────────────

class TestOpportunityScreener:

    def test_passing_candidate(self, active_feature_vector: FeatureVector) -> None:
        screener = OpportunityScreener()
        res = screener.screen(active_feature_vector, current_spread_pct=0.0005)
        assert res.passed is True
        assert res.stages_cleared == 3
        assert len(res.reasons) >= 3

    def test_penny_stock_rejected(self, active_feature_vector: FeatureVector) -> None:
        screener = OpportunityScreener(min_price=5.0)
        # Modify price to $3
        active_feature_vector.close_price = 3.0
        res = screener.screen(active_feature_vector)
        assert res.passed is False
        assert "below minimum" in res.reject_reason

    def test_wide_spread_rejected(self, active_feature_vector: FeatureVector) -> None:
        screener = OpportunityScreener(max_spread_pct=0.002)
        res = screener.screen(active_feature_vector, current_spread_pct=0.005)
        assert res.passed is False
        assert "Spread" in res.reject_reason

    def test_dormant_asset_rejected(self, active_feature_vector: FeatureVector) -> None:
        screener = OpportunityScreener(min_rvol=1.5, min_natr=1.0)
        active_feature_vector.rvol_20 = 0.8
        active_feature_vector.natr_14 = 0.5
        active_feature_vector.is_bb_squeeze = False

        res = screener.screen(active_feature_vector)
        assert res.passed is False
        assert "Asset dormant" in res.reject_reason


# ── 3. Opportunity Scorer Tests ──────────────────────────────────────

class TestOpportunityScorer:

    def test_scoring_candidate(self, active_feature_vector: FeatureVector) -> None:
        scored = OpportunityScorer.score(active_feature_vector)
        assert isinstance(scored, ScoredOpportunity)
        assert scored.symbol == "NVDA"
        assert scored.composite_score >= 70.0  # Strong setup
        assert scored.direction == "bullish"
        assert "trend" in scored.factor_scores
        assert "volume" in scored.factor_scores
        assert len(scored.reasons) >= 3

        d = scored.to_dict()
        assert d["symbol"] == "NVDA"
        assert "composite_score" in d


# ── 4. Universe Scanner Tests ────────────────────────────────────────

class TestUniverseScanner:

    def test_scan_from_features(self, active_feature_vector: FeatureVector) -> None:
        scanner = UniverseScanner()
        results = scanner.scan_from_features([active_feature_vector], min_score=50.0)
        assert len(results) == 1
        assert results[0].symbol == "NVDA"
        assert results[0].composite_score >= 50.0

    @pytest.mark.anyio
    async def test_scan_universe_data(self) -> None:
        np.random.seed(42)
        n = 60
        start = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
        times = [start + timedelta(days=i) for i in range(n)]

        def _make_df(drift: float, vol_mult: float) -> pd.DataFrame:
            closes = 100.0 + np.linspace(0, drift, n) + np.random.normal(0, 0.5, n)
            highs = closes + 1.0
            lows = closes - 1.0
            volumes = np.random.uniform(50000, 100000, n) * vol_mult
            df = pd.DataFrame({"time": times, "open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes})
            df.set_index("time", inplace=True)
            return df

        data_map = {
            "BULL_LEADER": _make_df(drift=30.0, vol_mult=2.5),  # High volume upward drift
            "FLAT_DORMANT": _make_df(drift=0.0, vol_mult=0.5),   # Low volume flat
            "BEAR_DOWNTREND": _make_df(drift=-25.0, vol_mult=2.0),
        }

        scanner = UniverseScanner()
        top_opps = await scanner.scan_universe_data(data_map, min_score=40.0, top_n=2)
        assert len(top_opps) > 0
        assert top_opps[0].composite_score >= top_opps[-1].composite_score

        # Executive report check
        report = UniverseScanner.generate_report(top_opps)
        assert "TOP QUANTITATIVE OPPORTUNITIES RANKING" in report
        assert "BULL_LEADER" in report or "BEAR_DOWNTREND" in report
