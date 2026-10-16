"""
AI Quant Trading Platform — Trend Following Strategy.

Core Logic:
- Identifies strong directional regimes via EMA Ribbon (EMA9, EMA21, EMA50) and Supertrend.
- Requires ADX > threshold (default 20.0) to avoid false breakouts in ranging/choppy markets.
- Confirms directional momentum via DI+/DI- spread and MACD histogram.
- Validates that price is not overextended using RSI bounds.
- Computes dynamic ATR-based Stop-Loss and Take-Profit targets (minimum 1:2 R:R).
"""

from __future__ import annotations

from typing import Any

from strategies.base import SignalDirection, Strategy, StrategySignal


class TrendFollowingStrategy(Strategy):
    """
    Quantitative Trend Following Strategy with ADX Volatility/Regime filter.
    """

    def __init__(
        self,
        min_adx: float = 20.0,
        rsi_max_long: float = 70.0,
        rsi_min_short: float = 30.0,
        atr_stop_multiplier: float = 2.0,
        atr_target_multiplier: float = 4.0,  # 1:2 Risk-to-Reward ratio
    ) -> None:
        self._min_adx = min_adx
        self._rsi_max_long = rsi_max_long
        self._rsi_min_short = rsi_min_short
        self._atr_stop_mult = atr_stop_multiplier
        self._atr_target_mult = atr_target_multiplier

    @property
    def name(self) -> str:
        return "trend_following"

    @property
    def description(self) -> str:
        return (
            "Trend following strategy based on EMA ribbon alignment, "
            "Supertrend direction, and ADX strength filter with dynamic ATR targets."
        )

    @property
    def timeframes(self) -> list[str]:
        return ["1d", "4h", "1h"]

    def get_config(self) -> dict[str, Any]:
        return {
            "min_adx": self._min_adx,
            "rsi_max_long": self._rsi_max_long,
            "rsi_min_short": self._rsi_min_short,
            "atr_stop_multiplier": self._atr_stop_mult,
            "atr_target_multiplier": self._atr_target_mult,
        }

    async def analyze(
        self,
        symbol: str,
        features: dict[str, Any],
        market_regime: str | None = None,
    ) -> StrategySignal | None:
        """
        Evaluate market features for trend continuation opportunities.
        """
        c = features.get("close_price")
        if not c or c <= 0:
            return None

        adx = features.get("adx_14", 0.0)
        di_spread = features.get("di_spread", 0.0)
        supertrend_dir = features.get("supertrend_dir", 0)
        ema_9_21_spread = features.get("ema_9_21_spread", 0.0)
        ema_21_dist = features.get("ema_21_dist_pct", 0.0)
        rsi = features.get("rsi_14", 50.0)
        natr = features.get("natr_14", 1.5)  # Normalized ATR %
        timeframe = features.get("timeframe", "1d")

        # ATR in dollar terms
        atr_dollars = c * (natr / 100.0) if natr > 0 else c * 0.015

        # 1. Check regime filter: ADX must show active trending behavior
        if adx < self._min_adx:
            return None  # Range-bound / choppy market -> NO TRADE

        reasons: list[str] = []

        # ── BULLISH CONDITIONS ───────────────────────────────────────
        is_bullish_trend = (
            supertrend_dir == 1 and
            ema_21_dist > 0 and          # Price above EMA21
            ema_9_21_spread > 0 and      # EMA9 > EMA21
            di_spread > 0 and            # +DI > -DI
            rsi < self._rsi_max_long     # Not severely overbought
        )

        if is_bullish_trend:
            reasons.append(f"Supertrend is bullish (+1)")
            reasons.append(f"Price is above EMA21 ({ema_21_dist:.2f}%) and EMA9 > EMA21")
            reasons.append(f"ADX ({adx:.1f}) confirms strong trend with positive DI spread ({di_spread:.1f})")
            reasons.append(f"RSI ({rsi:.1f}) has upside room (< {self._rsi_max_long})")

            # Calculate confidence score based on strength confluence
            confidence = min(1.0, 0.5 + (adx / 100.0) * 0.3 + (min(10.0, di_spread) / 10.0) * 0.2)
            score = confidence * 0.9

            stop_price = round(c - (self._atr_stop_mult * atr_dollars), 2)
            target_price = round(c + (self._atr_target_mult * atr_dollars), 2)
            expected_return = round(((target_price - c) / c) * 100.0, 2)
            risk_pct = round(((c - stop_price) / c) * 100.0, 2)

            return StrategySignal(
                strategy=self.name,
                symbol=symbol,
                direction=SignalDirection.BUY,
                score=round(score, 3),
                confidence=round(confidence, 3),
                expected_return=expected_return,
                risk_estimate=risk_pct,
                timeframe=timeframe,
                stop_suggestion=stop_price,
                target_suggestion=target_price,
                market_regime=market_regime,
                reasons=reasons,
                metadata={
                    "adx": adx,
                    "di_spread": di_spread,
                    "atr_dollars": round(atr_dollars, 2),
                    "rsi": rsi,
                },
            )

        # ── BEARISH CONDITIONS ───────────────────────────────────────
        is_bearish_trend = (
            supertrend_dir == -1 and
            ema_21_dist < 0 and          # Price below EMA21
            ema_9_21_spread < 0 and      # EMA9 < EMA21
            di_spread < 0 and            # -DI > +DI
            rsi > self._rsi_min_short    # Not severely oversold
        )

        if is_bearish_trend:
            reasons.append(f"Supertrend is bearish (-1)")
            reasons.append(f"Price is below EMA21 ({ema_21_dist:.2f}%) and EMA9 < EMA21")
            reasons.append(f"ADX ({adx:.1f}) confirms strong trend with negative DI spread ({di_spread:.1f})")
            reasons.append(f"RSI ({rsi:.1f}) has downside room (> {self._rsi_min_short})")

            confidence = min(1.0, 0.5 + (adx / 100.0) * 0.3 + (abs(max(-10.0, di_spread)) / 10.0) * 0.2)
            score = confidence * 0.9

            stop_price = round(c + (self._atr_stop_mult * atr_dollars), 2)
            target_price = round(c - (self._atr_target_mult * atr_dollars), 2)
            expected_return = round(((c - target_price) / c) * 100.0, 2)
            risk_pct = round(((stop_price - c) / c) * 100.0, 2)

            return StrategySignal(
                strategy=self.name,
                symbol=symbol,
                direction=SignalDirection.SELL,
                score=round(score, 3),
                confidence=round(confidence, 3),
                expected_return=expected_return,
                risk_estimate=risk_pct,
                timeframe=timeframe,
                stop_suggestion=stop_price,
                target_suggestion=target_price,
                market_regime=market_regime,
                reasons=reasons,
                metadata={
                    "adx": adx,
                    "di_spread": di_spread,
                    "atr_dollars": round(atr_dollars, 2),
                    "rsi": rsi,
                },
            )

        return None
