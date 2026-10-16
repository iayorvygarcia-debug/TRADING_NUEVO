"""
AI Quant Trading Platform — Breakout & Volatility Expansion Strategy.

Core Logic:
- Detects the explosive transition from low-volatility compression (Squeeze) to rapid expansion.
- Requires:
    1. Volume surge: RVOL >= 1.8 (volume at least 80% higher than 20-period average)
    2. Volatility trigger: Bollinger Band squeeze release or bandwidth expansion
    3. Institutional footprint: Fair Value Gap (FVG) or strong Chaikin Money Flow (CMF > 0.15)
    4. High-conviction close: Close near bar high (> 75% of candle range) for BUY, or near low (< 25%) for SELL
- Target: 3.0x ATR (high-velocity runners)
- Stop: Breakout bar low / 1.5x ATR
"""

from __future__ import annotations

from typing import Any

from strategies.base import SignalDirection, Strategy, StrategySignal


class BreakoutStrategy(Strategy):
    """
    Quantitative Volatility Expansion and Breakout Strategy.
    """

    def __init__(
        self,
        min_rvol: float = 1.8,
        min_cmf: float = 0.10,
        atr_stop_multiplier: float = 1.5,
        atr_target_multiplier: float = 3.5,
    ) -> None:
        self._min_rvol = min_rvol
        self._min_cmf = min_cmf
        self._atr_stop_mult = atr_stop_multiplier
        self._atr_target_mult = atr_target_multiplier

    @property
    def name(self) -> str:
        return "breakout"

    @property
    def description(self) -> str:
        return (
            "Breakout strategy targeting high-velocity volatility expansions after "
            "compression, confirmed by RVOL volume spikes and Fair Value Gaps."
        )

    @property
    def timeframes(self) -> list[str]:
        return ["5m", "15m", "1h", "1d"]

    def get_config(self) -> dict[str, Any]:
        return {
            "min_rvol": self._min_rvol,
            "min_cmf": self._min_cmf,
            "atr_stop_multiplier": self._atr_stop_mult,
            "atr_target_multiplier": self._atr_target_mult,
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

        rvol = features.get("rvol_20", 1.0)
        cmf = features.get("cmf_20", 0.0)
        has_fvg = features.get("has_fvg", False)
        is_squeeze = features.get("is_bb_squeeze", False)
        close_pos = features.get("close_position_in_bar", 0.5)
        natr = features.get("natr_14", 1.5)
        timeframe = features.get("timeframe", "1h")
        strongest_pattern = features.get("strongest_pattern", "")

        atr_dollars = c * (natr / 100.0) if natr > 0 else c * 0.015
        reasons: list[str] = []

        # ── BULLISH BREAKOUT ─────────────────────────────────────────
        # Criteria:
        # 1. High relative volume (institutional commitment)
        # 2. Strong close at top of the bar (> 75%)
        # 3. Positive money flow OR FVG imbalance OR Squeeze breakout
        is_bullish_breakout = (
            rvol >= self._min_rvol and
            close_pos >= 0.75 and
            (cmf >= self._min_cmf or has_fvg or is_squeeze)
        )

        if is_bullish_breakout:
            reasons.append(f"Institutional volume surge: RVOL is {rvol:.2f}x average")
            reasons.append(f"Strong bullish close at top {round((1.0 - close_pos) * 100)}% of candle range")
            if has_fvg:
                reasons.append("Unmitigated Fair Value Gap (demand imbalance) created")
            if cmf >= self._min_cmf:
                reasons.append(f"Positive institutional money flow: CMF is {cmf:.2f}")
            if is_squeeze:
                reasons.append("Volatility squeeze trigger (Bollinger Bands contracted)")

            confidence = min(0.95, 0.65 + (min(3.0, rvol) / 3.0) * 0.20 + (0.10 if has_fvg else 0.0))
            score = confidence * 0.90

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
                    "rvol": rvol,
                    "cmf": cmf,
                    "has_fvg": has_fvg,
                    "is_bb_squeeze": is_squeeze,
                    "strongest_pattern": strongest_pattern,
                },
            )

        # ── BEARISH BREAKDOWN ────────────────────────────────────────
        is_bearish_breakdown = (
            rvol >= self._min_rvol and
            close_pos <= 0.25 and
            (cmf <= -self._min_cmf or has_fvg or is_squeeze)
        )

        if is_bearish_breakdown:
            reasons.append(f"Institutional volume surge: RVOL is {rvol:.2f}x average")
            reasons.append(f"Strong bearish close at bottom {round(close_pos * 100)}% of candle range")
            if has_fvg:
                reasons.append("Bearish Fair Value Gap (supply imbalance) created")
            if cmf <= -self._min_cmf:
                reasons.append(f"Heavy institutional distribution: CMF is {cmf:.2f}")

            confidence = min(0.95, 0.65 + (min(3.0, rvol) / 3.0) * 0.20 + (0.10 if has_fvg else 0.0))
            score = confidence * 0.90

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
                    "rvol": rvol,
                    "cmf": cmf,
                    "has_fvg": has_fvg,
                    "is_bb_squeeze": is_squeeze,
                },
            )

        return None
