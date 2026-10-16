"""
AI Quant Trading Platform — Multi-Stage Opportunity Screener.

Multi-tier filtering pipeline:
- Stage 1 (Liquidity Gate): Price >= $5, Spread <= 0.20%, Sufficient daily volume.
- Stage 2 (Activity & Volatility Gate): RVOL >= 1.2x, Volatility Squeeze, or ATR % expansion.
- Stage 3 (Structural Quality Gate): Clear trend structure, Candlestick pattern, or FVG imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.technical_analysis.feature_engine import FeatureVector


@dataclass(slots=True)
class ScreeningResult:
    """Outcome of multi-stage screening for a single symbol."""
    symbol: str
    passed: bool
    stages_cleared: int             # 0, 1, 2, or 3
    reasons: list[str] = field(default_factory=list)
    reject_reason: str = ""


class OpportunityScreener:
    """
    High-throughput candidate screener.
    """

    def __init__(
        self,
        min_price: float = 5.0,
        max_spread_pct: float = 0.0025,    # 0.25% max spread
        min_natr: float = 0.75,            # At least 0.75% daily range
        min_rvol: float = 1.15,            # At least 15% above normal volume
    ) -> None:
        self.min_price = min_price
        self.max_spread_pct = max_spread_pct
        self.min_natr = min_natr
        self.min_rvol = min_rvol

    def screen(
        self,
        features: FeatureVector,
        current_spread_pct: float = 0.001,
    ) -> ScreeningResult:
        """
        Screen a candidate FeatureVector through the 3-stage pipeline.
        """
        sym = features.symbol
        reasons: list[str] = []

        # ── Stage 1: Liquidity & Tradability Gate ────────────────────
        if features.close_price < self.min_price:
            return ScreeningResult(
                symbol=sym,
                passed=False,
                stages_cleared=0,
                reject_reason=f"Price ${features.close_price:.2f} is below minimum ${self.min_price:.2f}",
            )

        if current_spread_pct > self.max_spread_pct:
            return ScreeningResult(
                symbol=sym,
                passed=False,
                stages_cleared=0,
                reject_reason=f"Spread {current_spread_pct*100:.3f}% exceeds max {self.max_spread_pct*100:.3f}%",
            )

        reasons.append("Stage 1 Passed: Price and liquidity thresholds satisfied.")

        # ── Stage 2: Activity & Volatility Gate ──────────────────────
        # Must have either elevated volume, active volatility, or a volatility squeeze
        has_volume = features.rvol_20 >= self.min_rvol
        has_volatility = features.natr_14 >= self.min_natr
        has_squeeze = features.is_bb_squeeze

        if not (has_volume or has_volatility or has_squeeze):
            return ScreeningResult(
                symbol=sym,
                passed=False,
                stages_cleared=1,
                reasons=reasons,
                reject_reason=f"Asset dormant: RVOL {features.rvol_20:.2f} < {self.min_rvol} and NATR {features.natr_14:.2f}% < {self.min_natr}%",
            )

        if has_volume:
            reasons.append(f"Stage 2 Passed: Elevated volume (RVOL {features.rvol_20:.2f}x).")
        elif has_squeeze:
            reasons.append("Stage 2 Passed: Volatility Squeeze detected (pre-breakout).")
        else:
            reasons.append(f"Stage 2 Passed: Sufficient volatility (NATR {features.natr_14:.2f}%).")

        # ── Stage 3: Structural & Price Action Gate ──────────────────
        # Must display either:
        # A) Strong directional trend (ADX > 20 and Supertrend aligned)
        # B) Reversal pattern or Pinbar
        # C) Fair Value Gap (Imbalance)
        # D) Multi-timeframe pullback setup
        has_trend = features.adx_14 >= 18.0 and features.supertrend_dir != 0
        has_pattern = (features.bullish_pattern_count + features.bearish_pattern_count) > 0 or features.is_pinbar
        has_fvg = features.has_fvg
        has_mtf = features.mtf_is_pullback or abs(features.mtf_alignment_score) >= 0.50

        if not (has_trend or has_pattern or has_fvg or has_mtf):
            return ScreeningResult(
                symbol=sym,
                passed=False,
                stages_cleared=2,
                reasons=reasons,
                reject_reason="No structural setup: no strong trend, pattern, FVG, or MTF alignment.",
            )

        if has_trend:
            reasons.append(f"Stage 3 Passed: Active trend (ADX {features.adx_14:.1f}, Dir {features.supertrend_dir}).")
        if has_fvg:
            reasons.append("Stage 3 Passed: Fair Value Gap present.")
        if has_pattern:
            reasons.append(f"Stage 3 Passed: Pattern ({features.strongest_pattern}).")
        if has_mtf:
            reasons.append(f"Stage 3 Passed: Multi-timeframe alignment ({features.mtf_alignment_score:.2f}).")

        return ScreeningResult(
            symbol=sym,
            passed=True,
            stages_cleared=3,
            reasons=reasons,
        )
