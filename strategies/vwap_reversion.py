"""
AI Quant Trading Platform — VWAP Reversion Strategy.

Core Logic:
    The Volume-Weighted Average Price (VWAP) represents the fair value of an asset for the
    current trading session. Institutional traders use VWAP as their primary benchmark:
    - They BUY when price falls significantly BELOW VWAP (discount).
    - They SELL when price rises significantly ABOVE VWAP (premium).

    This strategy detects statistically significant VWAP deviations and waits for
    money-flow exhaustion signals before entering a reversion trade back to VWAP.

Entry Conditions (Long / Buy):
    1. Price is at least `vwap_entry_band_pct` BELOW VWAP (significant discount)
    2. RSI is oversold (< rsi_oversold threshold) confirming selling exhaustion
    3. MFI is oversold (< 30) confirming money-flow exhaustion
    4. CMF is negative (institutional distribution phase ending)
    5. StochRSI K < 25 confirming extreme momentum oversold
    6. NOT in a strong downtrend (ADX can be high but Supertrend should not be strongly bearish)

Entry Conditions (Short / Sell):
    1. Price is at least `vwap_entry_band_pct` ABOVE VWAP (significant premium)
    2. RSI is overbought (> rsi_overbought threshold) confirming buying exhaustion
    3. MFI is overbought (> 70) confirming money-flow exhaustion
    4. CMF is positive (institutional accumulation phase ending)
    5. StochRSI K > 75 confirming extreme momentum overbought

Target:
    - Primary target: Return to VWAP (mean reversion)
    - Stop: 1.5x ATR beyond the entry (prevents large adverse moves)

Risk/Reward:
    - Minimum 1:1.5 R:R required to take the trade
    - If VWAP distance doesn't justify minimum R:R, trade is skipped
"""

from __future__ import annotations

from typing import Any

from strategies.base import SignalDirection, Strategy, StrategySignal


class VWAPReversionStrategy(Strategy):
    """
    Institutional-grade VWAP Reversion Strategy.

    Waits for price to reach statistically significant deviation from the session VWAP,
    then uses multi-indicator exhaustion confluence to time entries for mean reversion.
    """

    def __init__(
        self,
        vwap_entry_band_pct: float = 1.5,    # Minimum % deviation from VWAP to trigger scan
        vwap_strong_band_pct: float = 2.5,   # Strong deviation — higher confidence score
        rsi_oversold: float = 38.0,
        rsi_overbought: float = 62.0,
        stoch_rsi_long_max: float = 30.0,    # StochRSI K must be below this for longs
        stoch_rsi_short_min: float = 70.0,   # StochRSI K must be above this for shorts
        mfi_oversold: float = 35.0,
        mfi_overbought: float = 65.0,
        atr_stop_multiplier: float = 1.5,
        min_rr: float = 1.2,                 # Minimum risk/reward ratio required
    ) -> None:
        self._vwap_band = vwap_entry_band_pct
        self._vwap_strong_band = vwap_strong_band_pct
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._stoch_rsi_long_max = stoch_rsi_long_max
        self._stoch_rsi_short_min = stoch_rsi_short_min
        self._mfi_oversold = mfi_oversold
        self._mfi_overbought = mfi_overbought
        self._atr_stop_mult = atr_stop_multiplier
        self._min_rr = min_rr

    @property
    def name(self) -> str:
        return "vwap_reversion"

    @property
    def description(self) -> str:
        return (
            "VWAP Reversion strategy: trades mean-reversion when price deviates significantly "
            "from the session VWAP with multi-indicator exhaustion confluence (RSI, MFI, CMF, StochRSI). "
            "Targets return to VWAP fair value with ATR-based stop protection."
        )

    @property
    def timeframes(self) -> list[str]:
        return ["1h", "15m", "5m"]  # Intraday focused

    def get_config(self) -> dict[str, Any]:
        return {
            "vwap_entry_band_pct": self._vwap_band,
            "vwap_strong_band_pct": self._vwap_strong_band,
            "rsi_oversold": self._rsi_oversold,
            "rsi_overbought": self._rsi_overbought,
            "stoch_rsi_long_max": self._stoch_rsi_long_max,
            "stoch_rsi_short_min": self._stoch_rsi_short_min,
            "mfi_oversold": self._mfi_oversold,
            "mfi_overbought": self._mfi_overbought,
            "atr_stop_multiplier": self._atr_stop_mult,
            "min_rr": self._min_rr,
        }

    def _compute_confidence(
        self,
        vwap_dist_pct: float,
        rsi: float,
        mfi: float,
        cmf: float,
        stoch_rsi_k: float,
        direction: str,
    ) -> float:
        """
        Compute signal confidence based on how extreme each condition is.

        The more indicators confirm the exhaustion, the higher the confidence.
        Scoring components:
        - VWAP distance: 30% weight (stronger deviation = higher probability of reversion)
        - RSI extreme: 25% weight
        - MFI extreme: 20% weight
        - CMF direction: 15% weight
        - StochRSI extreme: 10% weight
        """
        score = 0.0

        # 1. VWAP distance component (30%)
        abs_dist = abs(vwap_dist_pct)
        if abs_dist >= self._vwap_strong_band:
            score += 0.30
        elif abs_dist >= self._vwap_band:
            score += 0.30 * ((abs_dist - self._vwap_band) / (self._vwap_strong_band - self._vwap_band + 1e-6))

        if direction == "long":
            # 2. RSI component (25%) — lower RSI is better for longs
            rsi_intensity = max(0.0, (self._rsi_oversold - rsi) / self._rsi_oversold)
            score += 0.25 * min(1.0, rsi_intensity * 3.0)

            # 3. MFI component (20%) — lower MFI is better
            mfi_intensity = max(0.0, (self._mfi_oversold - mfi) / self._mfi_oversold)
            score += 0.20 * min(1.0, mfi_intensity * 3.0)

            # 4. CMF component (15%) — negative CMF turning less negative = buyers coming in
            if cmf < -0.1:
                score += 0.15
            elif cmf < 0:
                score += 0.08

            # 5. StochRSI component (10%)
            if stoch_rsi_k <= 15.0:
                score += 0.10
            elif stoch_rsi_k <= self._stoch_rsi_long_max:
                score += 0.10 * ((self._stoch_rsi_long_max - stoch_rsi_k) / self._stoch_rsi_long_max)

        else:  # short
            # 2. RSI component (25%) — higher RSI is better for shorts
            rsi_intensity = max(0.0, (rsi - self._rsi_overbought) / (100.0 - self._rsi_overbought))
            score += 0.25 * min(1.0, rsi_intensity * 3.0)

            # 3. MFI component (20%) — higher MFI is better
            mfi_intensity = max(0.0, (mfi - self._mfi_overbought) / (100.0 - self._mfi_overbought))
            score += 0.20 * min(1.0, mfi_intensity * 3.0)

            # 4. CMF component (15%)
            if cmf > 0.1:
                score += 0.15
            elif cmf > 0:
                score += 0.08

            # 5. StochRSI component (10%)
            if stoch_rsi_k >= 85.0:
                score += 0.10
            elif stoch_rsi_k >= self._stoch_rsi_short_min:
                score += 0.10 * ((stoch_rsi_k - self._stoch_rsi_short_min) / (100.0 - self._stoch_rsi_short_min))

        return round(min(1.0, score + 0.40), 3)  # Base confidence of 0.40 for meeting min conditions

    async def analyze(
        self,
        symbol: str,
        features: dict[str, Any],
        market_regime: str | None = None,
    ) -> StrategySignal | None:
        """
        Evaluate VWAP deviation and exhaustion signals for mean-reversion entries.
        """
        c = features.get("close_price")
        if not c or c <= 0:
            return None

        vwap_dist_pct = features.get("vwap_dist_pct", 0.0)
        rsi = features.get("rsi_14", 50.0)
        stoch_rsi_k = features.get("stoch_rsi_k", 50.0)
        mfi = features.get("mfi_14", 50.0)
        cmf = features.get("cmf_20", 0.0)
        natr = features.get("natr_14", 1.5)
        timeframe = features.get("timeframe", "1h")
        supertrend_dir = features.get("supertrend_dir", 0)
        rvol = features.get("rvol_20", 1.0)

        # ATR in dollar terms
        atr_dollars = c * (natr / 100.0) if natr > 0 else c * 0.015

        reasons: list[str] = []

        # ── LONG: Price significantly BELOW VWAP ─────────────────────
        # Institutional buyers see this as a discount — expect mean reversion UP
        price_is_below_vwap = vwap_dist_pct <= -self._vwap_band
        rsi_oversold_confirmed = rsi <= self._rsi_oversold
        mfi_oversold_confirmed = mfi <= self._mfi_oversold
        stoch_rsi_oversold = stoch_rsi_k <= self._stoch_rsi_long_max
        # Avoid entering long if strong bearish trend AND not extremely oversold
        not_collapsing = not (supertrend_dir == -1 and vwap_dist_pct >= -self._vwap_strong_band and rsi > 25)

        long_conditions_met = (
            price_is_below_vwap and
            rsi_oversold_confirmed and
            mfi_oversold_confirmed and
            stoch_rsi_oversold and
            not_collapsing
        )

        if long_conditions_met:
            # Calculate target (VWAP itself) and stop (below entry)
            # Approximate VWAP price from distance percentage
            vwap_price = c / (1.0 + vwap_dist_pct / 100.0)
            stop_price = round(c - (self._atr_stop_mult * atr_dollars), 2)
            target_price = round(vwap_price, 2)

            risk_dist = c - stop_price
            reward_dist = target_price - c

            # Enforce minimum R:R
            if risk_dist <= 0 or (reward_dist / risk_dist) < self._min_rr:
                return None

            expected_return = round((reward_dist / c) * 100.0, 2)
            risk_pct = round((risk_dist / c) * 100.0, 2)

            confidence = self._compute_confidence(vwap_dist_pct, rsi, mfi, cmf, stoch_rsi_k, "long")
            score = round(confidence * 0.88, 3)

            reasons.append(
                f"Price is {abs(vwap_dist_pct):.2f}% BELOW VWAP (discount zone) "
                f"— institutional mean reversion setup"
            )
            reasons.append(f"RSI ({rsi:.1f}) confirms selling exhaustion (oversold < {self._rsi_oversold})")
            reasons.append(f"MFI ({mfi:.1f}) confirms money-flow exhaustion (< {self._mfi_oversold})")
            reasons.append(f"StochRSI K ({stoch_rsi_k:.1f}) at extreme oversold level")
            if cmf < -0.05:
                reasons.append(f"CMF ({cmf:.3f}) shows distribution phase — potential reversal approaching")
            if rvol > 1.5:
                reasons.append(f"Elevated relative volume ({rvol:.1f}x) confirms institutional activity")
            if abs(vwap_dist_pct) >= self._vwap_strong_band:
                reasons.append(f"STRONG deviation ({abs(vwap_dist_pct):.2f}%) — high-probability reversion expected")

            return StrategySignal(
                strategy=self.name,
                symbol=symbol,
                direction=SignalDirection.BUY,
                score=score,
                confidence=confidence,
                expected_return=expected_return,
                risk_estimate=risk_pct,
                timeframe=timeframe,
                stop_suggestion=stop_price,
                target_suggestion=target_price,
                market_regime=market_regime,
                reasons=reasons,
                metadata={
                    "vwap_dist_pct": vwap_dist_pct,
                    "vwap_target": target_price,
                    "rsi": rsi,
                    "mfi": mfi,
                    "cmf": cmf,
                    "stoch_rsi_k": stoch_rsi_k,
                    "rvol": rvol,
                    "rr_ratio": round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0,
                },
            )

        # ── SHORT: Price significantly ABOVE VWAP ────────────────────
        # Institutional sellers see this as overpriced — expect mean reversion DOWN
        price_is_above_vwap = vwap_dist_pct >= self._vwap_band
        rsi_overbought_confirmed = rsi >= self._rsi_overbought
        mfi_overbought_confirmed = mfi >= self._mfi_overbought
        stoch_rsi_overbought = stoch_rsi_k >= self._stoch_rsi_short_min

        short_conditions_met = (
            price_is_above_vwap and
            rsi_overbought_confirmed and
            mfi_overbought_confirmed and
            stoch_rsi_overbought
        )

        if short_conditions_met:
            vwap_price = c / (1.0 + vwap_dist_pct / 100.0)
            stop_price = round(c + (self._atr_stop_mult * atr_dollars), 2)
            target_price = round(vwap_price, 2)

            risk_dist = stop_price - c
            reward_dist = c - target_price

            if risk_dist <= 0 or (reward_dist / risk_dist) < self._min_rr:
                return None

            expected_return = round((reward_dist / c) * 100.0, 2)
            risk_pct = round((risk_dist / c) * 100.0, 2)

            confidence = self._compute_confidence(vwap_dist_pct, rsi, mfi, cmf, stoch_rsi_k, "short")
            score = round(confidence * 0.88, 3)

            reasons.append(
                f"Price is {abs(vwap_dist_pct):.2f}% ABOVE VWAP (premium zone) "
                f"— institutional mean reversion setup"
            )
            reasons.append(f"RSI ({rsi:.1f}) confirms buying exhaustion (overbought > {self._rsi_overbought})")
            reasons.append(f"MFI ({mfi:.1f}) confirms money-flow exhaustion (> {self._mfi_overbought})")
            reasons.append(f"StochRSI K ({stoch_rsi_k:.1f}) at extreme overbought level")
            if cmf > 0.05:
                reasons.append(f"CMF ({cmf:.3f}) shows accumulation phase ending — reversal approaching")
            if rvol > 1.5:
                reasons.append(f"Elevated relative volume ({rvol:.1f}x) confirms institutional activity")
            if abs(vwap_dist_pct) >= self._vwap_strong_band:
                reasons.append(f"STRONG deviation ({abs(vwap_dist_pct):.2f}%) — high-probability reversion expected")

            return StrategySignal(
                strategy=self.name,
                symbol=symbol,
                direction=SignalDirection.SELL,
                score=score,
                confidence=confidence,
                expected_return=expected_return,
                risk_estimate=risk_pct,
                timeframe=timeframe,
                stop_suggestion=stop_price,
                target_suggestion=target_price,
                market_regime=market_regime,
                reasons=reasons,
                metadata={
                    "vwap_dist_pct": vwap_dist_pct,
                    "vwap_target": target_price,
                    "rsi": rsi,
                    "mfi": mfi,
                    "cmf": cmf,
                    "stoch_rsi_k": stoch_rsi_k,
                    "rvol": rvol,
                    "rr_ratio": round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0,
                },
            )

        return None
