"""
Unit Tests for Quantitative Strategies and Signal Combiner.

Validates:
1. TrendFollowingStrategy logic & filters
2. MeanReversionStrategy logic & filters
3. BreakoutStrategy logic & filters
4. SignalCombiner conflict resolution & ensemble scoring
"""

from __future__ import annotations

import pytest

from strategies.base import SignalDirection, StrategySignal
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from services.strategy_engine.signal_combiner import SignalCombiner, CombinedSignal


class TestTrendFollowingStrategy:

    @pytest.fixture
    def strategy(self) -> TrendFollowingStrategy:
        return TrendFollowingStrategy(min_adx=20.0)

    @pytest.mark.anyio
    async def test_bullish_signal(self, strategy: TrendFollowingStrategy) -> None:
        features = {
            "close_price": 150.0,
            "supertrend_dir": 1,
            "ema_21_dist_pct": 2.5,
            "ema_9_21_spread": 1.2,
            "adx_14": 28.5,
            "di_spread": 8.0,
            "rsi_14": 58.0,
            "natr_14": 1.5,
            "timeframe": "1d",
        }
        signal = await strategy.analyze("AAPL", features)
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.score > 0.6
        assert signal.stop_suggestion is not None and signal.stop_suggestion < 150.0
        assert signal.target_suggestion is not None and signal.target_suggestion > 150.0
        assert len(signal.reasons) >= 3

    @pytest.mark.anyio
    async def test_adx_filter_rejects_choppy_market(self, strategy: TrendFollowingStrategy) -> None:
        # Same bullish setup but ADX is low (14.0 < 20.0)
        features = {
            "close_price": 150.0,
            "supertrend_dir": 1,
            "ema_21_dist_pct": 2.5,
            "ema_9_21_spread": 1.2,
            "adx_14": 14.0,  # Below threshold
            "di_spread": 2.0,
            "rsi_14": 52.0,
            "natr_14": 1.5,
        }
        signal = await strategy.analyze("AAPL", features)
        assert signal is None  # Must NOT trade in choppy markets!

    @pytest.mark.anyio
    async def test_bearish_signal(self, strategy: TrendFollowingStrategy) -> None:
        features = {
            "close_price": 150.0,
            "supertrend_dir": -1,
            "ema_21_dist_pct": -2.5,
            "ema_9_21_spread": -1.2,
            "adx_14": 30.0,
            "di_spread": -8.0,
            "rsi_14": 42.0,
            "natr_14": 1.5,
            "timeframe": "1d",
        }
        signal = await strategy.analyze("AAPL", features)
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.stop_suggestion is not None and signal.stop_suggestion > 150.0
        assert signal.target_suggestion is not None and signal.target_suggestion < 150.0


class TestMeanReversionStrategy:

    @pytest.fixture
    def strategy(self) -> MeanReversionStrategy:
        return MeanReversionStrategy()

    @pytest.mark.anyio
    async def test_oversold_bounce_signal(self, strategy: MeanReversionStrategy) -> None:
        features = {
            "close_price": 100.0,
            "rsi_14": 28.0,
            "bb_percent": 0.08,  # Below lower band
            "stoch_rsi_k": 12.0,
            "bullish_pattern_count": 1,
            "is_pinbar": True,
            "strongest_pattern": "hammer",
            "natr_14": 2.0,
            "timeframe": "1h",
        }
        signal = await strategy.analyze("TSLA", features)
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.stop_suggestion is not None and signal.stop_suggestion < 100.0
        assert signal.target_suggestion is not None and signal.target_suggestion > 100.0

    @pytest.mark.anyio
    async def test_neutral_market_returns_none(self, strategy: MeanReversionStrategy) -> None:
        features = {
            "close_price": 100.0,
            "rsi_14": 50.0,
            "bb_percent": 0.50,
            "stoch_rsi_k": 50.0,
            "bullish_pattern_count": 0,
            "bearish_pattern_count": 0,
        }
        signal = await strategy.analyze("TSLA", features)
        assert signal is None


class TestBreakoutStrategy:

    @pytest.fixture
    def strategy(self) -> BreakoutStrategy:
        return BreakoutStrategy(min_rvol=1.8)

    @pytest.mark.anyio
    async def test_bullish_breakout(self, strategy: BreakoutStrategy) -> None:
        features = {
            "close_price": 200.0,
            "rvol_20": 2.4,                  # Big volume surge
            "close_position_in_bar": 0.88,   # Closed near top
            "has_fvg": True,
            "cmf_20": 0.18,
            "is_bb_squeeze": False,
            "natr_14": 1.8,
            "timeframe": "15m",
        }
        signal = await strategy.analyze("NVDA", features)
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence >= 0.80

    @pytest.mark.anyio
    async def test_low_volume_rejected(self, strategy: BreakoutStrategy) -> None:
        features = {
            "close_price": 200.0,
            "rvol_20": 1.1,                  # Low volume -> rejected
            "close_position_in_bar": 0.85,
            "has_fvg": True,
            "cmf_20": 0.12,
        }
        signal = await strategy.analyze("NVDA", features)
        assert signal is None


class TestSignalCombiner:

    def test_conflict_resolves_to_hold(self) -> None:
        combiner = SignalCombiner(resolve_conflicts_to_hold=True)
        sig1 = StrategySignal(
            strategy="trend_following",
            symbol="AAPL",
            direction=SignalDirection.BUY,
            score=0.85,
            confidence=0.80,
        )
        sig2 = StrategySignal(
            strategy="mean_reversion",
            symbol="AAPL",
            direction=SignalDirection.SELL,
            score=0.80,
            confidence=0.75,
        )
        combined = combiner.combine("AAPL", [sig1, sig2])
        assert combined.direction == SignalDirection.HOLD
        assert not combined.is_actionable
        assert "Conflict detected" in combined.reasons[0]

    def test_agreement_produces_approved_signal(self) -> None:
        combiner = SignalCombiner(min_composite_score=0.65)
        sig1 = StrategySignal(
            strategy="trend_following",
            symbol="MSFT",
            direction=SignalDirection.BUY,
            score=0.85,
            confidence=0.80,
            stop_suggestion=420.0,
            target_suggestion=460.0,
        )
        sig2 = StrategySignal(
            strategy="breakout",
            symbol="MSFT",
            direction=SignalDirection.BUY,
            score=0.88,
            confidence=0.85,
            stop_suggestion=425.0,
            target_suggestion=465.0,
        )
        combined = combiner.combine("MSFT", [sig1, sig2])
        assert combined.direction == SignalDirection.BUY
        assert combined.is_approved
        assert combined.composite_score > 0.80
        # For BUY: conservative stop is the higher stop (425.0 > 420.0)
        assert combined.suggested_stop == 425.0
