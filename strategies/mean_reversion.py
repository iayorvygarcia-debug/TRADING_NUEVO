"""
AI Quant Trading Platform — Mean Reversion / Pullback Strategy.

Core Logic:
- Identifies temporary price exhausts and pullbacks where price deviates excessively from fair value.
- Triggers when:
    1. Multi-timeframe confluence (Higher TF trending, Lower TF in oversold dip/pullback)
    2. Bollinger %B extremes (< 0.15 or > 0.85) combined with RSI & StochRSI extremes
    3. Confirmatory price action reversal (Pinbar, Hammer, Engulfing pattern)
- Target: Return to mean (EMA21 / Bollinger middle band)
- Stop: Beyond the recent swing extreme (1.5x ATR)
"""

from __future__ import annotations

from typing import Any

from strategies.base import SignalDirection, Strategy, StrategySignal


class MeanReversionStrategy(Strategy):
    """
    Quantitative Mean Reversion Strategy combining statistical bands, momentum extremes,
    and candlestick reversal confirmations.
    """

    def __init__(
        self,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        bb_percent_oversold: float = 0.15,
        bb_percent_overbought: float = 0.85,
        atr_stop_multiplier: float = 1.5,
    ) -> None:
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._bb_pct_oversold = bb_percent_oversold
        self._bb_pct_overbought = bb_percent_overbought
        self._atr_stop_mult = atr_stop_multiplier

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def description(self) -> str:
        return (
            "Mean reversion strategy exploiting statistical extremes in Bollinger Bands and RSI, "
            "confirmed by price action reversal patterns and multi-timeframe pullback confluence."
        )

    @property
    def timeframes(self) -> list[str]:
        return ["15m", "1h", "1d"]

    def get_config(self) -> dict[str, Any]:
        return {
            "rsi_oversold": self._rsi_oversold,
            "rsi_overbought": self._rsi_overbought,
            "bb_percent_oversold": self._bb_pct_oversold,
            "bb_percent_overbought": self._bb_pct_overbought,
            "atr_stop_multiplier": self._atr_stop_mult,
        }

    async def analyze(
        self,
        symbol: str,
        features: dict[str, Any],
        market_regime: str | None = None,
    ) -> StrategySignal | None:
        c = features.get("close_price")
        if not c or c <= 0:
            return None

        rsi = features.get("rsi_14", 50.0)
        bb_percent = features.get("bb_percent", 0.5)
        stoch_rsi_k = features.get("stoch_rsi_k", 50.0)
        natr = features.get("natr_14", 1.5)
        timeframe = features.get("timeframe", "1h")

        is_pullback = features.get("mtf_is_pullback", False)
        mtf_score = features.get("mtf_alignment_score", 0.0)
        bullish_patterns = features.get("bullish_pattern_count", 0)
        bearish_patterns = features.get("bearish_pattern_count", 0)
        is_pinbar = features.get("is_pinbar", False)
        strongest_pattern = features.get("strongest_pattern", "")

        atr_dollars = c * (natr / 100.0) if natr > 0 else c * 0.015
        reasons: list[str] = []

        # ── BUY REVERSION (OVERSOLD DIP) ─────────────────────────────
        # Conditions:
        # A) Extreme Bollinger & RSI oversold + Reversal candle, OR
        # B) Multi-timeframe trend is strongly bullish and price is in a pullback
        is_extreme_oversold = (
            bb_percent <= self._bb_pct_oversold and
            rsi <= self._rsi_oversold and
            stoch_rsi_k <= 25.0 and
            (bullish_patterns > 0 or is_pinbar)
        )

        is_high_prob_pullback = (
            is_pullback and
            mtf_score > 0.3 and
            rsi <= 45.0 and
            (bullish_patterns > 0 or is_pinbar or bb_percent <= 0.30)
        )

        if is_extreme_oversold or is_high_prob_pullback:
            if is_extreme_oversold:
                reasons.append(f"Bollinger %B ({bb_percent:.2f}) and RSI ({rsi:.1f}) hit extreme oversold levels")
            if is_high_prob_pullback:
                reasons.append(f"Multi-timeframe pullback confirmed (Macro Trend + LTF Dip, score {mtf_score:.2f})")
            if bullish_patterns > 0:
                reasons.append(f"Bullish reversal pattern detected: {strongest_pattern}")
            if is_pinbar:
                reasons.append("Reversal pinbar rejecting lower prices")

            confidence = 0.75 if is_high_prob_pullback else 0.70
            if bullish_patterns > 0 and is_pinbar:
                confidence = min(0.95, confidence + 0.15)

            stop_price = round(c - (self._atr_stop_mult * atr_dollars), 2)
            # Target mean is at least 2x risk distance
            risk_dist = c - stop_price
            target_price = round(c + (2.0 * risk_dist), 2)
            expected_return = round(((target_price - c) / c) * 100.0, 2)
            risk_pct = round(((c - stop_price) / c) * 100.0, 2)

            return StrategySignal(
                strategy=self.name,
                symbol=symbol,
                direction=SignalDirection.BUY,
                score=round(confidence * 0.85, 3),
                confidence=round(confidence, 3),
                expected_return=expected_return,
                risk_estimate=risk_pct,
                timeframe=timeframe,
                stop_suggestion=stop_price,
                target_suggestion=target_price,
                market_regime=market_regime,
                reasons=reasons,
                metadata={
                    "bb_percent": bb_percent,
                    "rsi": rsi,
                    "is_pullback": is_pullback,
                    "strongest_pattern": strongest_pattern,
                },
            )

        # ── SELL REVERSION (OVERBOUGHT RALLY) ────────────────────────
        is_extreme_overbought = (
            bb_percent >= self._bb_pct_overbought and
            rsi >= self._rsi_overbought and
            stoch_rsi_k >= 75.0 and
            (bearish_patterns > 0 or is_pinbar)
        )

        if is_extreme_overbought:
            reasons.append(f"Bollinger %B ({bb_percent:.2f}) and RSI ({rsi:.1f}) hit extreme overbought levels")
            if bearish_patterns > 0:
                reasons.append(f"Bearish reversal pattern detected: {strongest_pattern}")
            if is_pinbar:
                reasons.append("Reversal pinbar rejecting higher prices")

            confidence = 0.72
            if bearish_patterns > 0 and is_pinbar:
                confidence = min(0.92, confidence + 0.15)

            stop_price = round(c + (self._atr_stop_mult * atr_dollars), 2)
            risk_dist = stop_price - c
            target_price = round(c - (2.0 * risk_dist), 2)
            expected_return = round(((c - target_price) / c) * 100.0, 2)
            risk_pct = round(((stop_price - c) / c) * 100.0, 2)

            return StrategySignal(
                strategy=self.name,
                symbol=symbol,
                direction=SignalDirection.SELL,
                score=round(confidence * 0.85, 3),
                confidence=round(confidence, 3),
                expected_return=expected_return,
                risk_estimate=risk_pct,
                timeframe=timeframe,
                stop_suggestion=stop_price,
                target_suggestion=target_price,
                market_regime=market_regime,
                reasons=reasons,
                metadata={
                    "bb_percent": bb_percent,
                    "rsi": rsi,
                    "strongest_pattern": strongest_pattern,
                },
            )

        return None
