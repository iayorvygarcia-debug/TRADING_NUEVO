"""
AI Quant Trading Platform — Opportunity Scorer.

Computes a multi-factor quantitative composite score (0 to 100) for every candidate:
1. Trend Factor (25%): EMA alignment, Supertrend, ADX strength.
2. Momentum Factor (25%): RSI location, MACD histogram momentum, Stochastic RSI.
3. Volume & Flow Factor (25%): RVOL, Chaikin Money Flow (CMF), MFI.
4. Price Action & Catalyst Factor (25%): Candlestick patterns, FVGs, Squeeze setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.technical_analysis.feature_engine import FeatureVector


@dataclass(slots=True)
class ScoredOpportunity:
    """Scored and categorized trading candidate."""
    symbol: str
    composite_score: float             # 0.0 to 100.0
    direction: str                     # "bullish", "bearish", "neutral"
    setup_type: str                    # "TREND_CONTINUATION", "VOLATILITY_BREAKOUT", "MEAN_REVERSION", "LIQUIDITY_IMBALANCE"
    factor_scores: dict[str, float]    # Individual factor breakdowns
    reasons: list[str] = field(default_factory=list)
    features: FeatureVector | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "composite_score": round(self.composite_score, 1),
            "direction": self.direction,
            "setup_type": self.setup_type,
            "factor_scores": {k: round(v, 1) for k, v in self.factor_scores.items()},
            "reasons": self.reasons,
        }


class OpportunityScorer:
    """
    Quantitative 4-Factor Opportunity Scorer.
    """

    @classmethod
    def score(cls, f: FeatureVector) -> ScoredOpportunity:
        """
        Evaluate a FeatureVector and generate a comprehensive ScoredOpportunity.
        """
        reasons: list[str] = []

        # ── 1. Trend Factor (Max 25 pts) ─────────────────────────────
        trend_score = 0.0
        bullish_bias_pts = 0
        bearish_bias_pts = 0

        # Supertrend
        if f.supertrend_dir == 1:
            trend_score += 8.0
            bullish_bias_pts += 2
        elif f.supertrend_dir == -1:
            trend_score += 8.0
            bearish_bias_pts += 2

        # EMA Ribbon
        if f.ema_9_21_spread > 0.5:
            trend_score += 7.0
            bullish_bias_pts += 2
        elif f.ema_9_21_spread < -0.5:
            trend_score += 7.0
            bearish_bias_pts += 2

        # ADX Trend Strength
        if f.adx_14 >= 25.0:
            trend_score += 10.0
            reasons.append(f"Strong directional trend (ADX {f.adx_14:.1f})")
        elif f.adx_14 >= 18.0:
            trend_score += 5.0

        # ── 2. Momentum Factor (Max 25 pts) ──────────────────────────
        mom_score = 0.0
        if 50.0 <= f.rsi_14 <= 68.0:
            mom_score += 10.0
            bullish_bias_pts += 2
        elif 32.0 <= f.rsi_14 <= 50.0:
            mom_score += 10.0
            bearish_bias_pts += 2
        elif f.rsi_14 < 30.0:  # Oversold pullback candidate
            mom_score += 8.0
            bullish_bias_pts += 3
        elif f.rsi_14 > 70.0:  # Overbought pullback candidate
            mom_score += 8.0
            bearish_bias_pts += 3

        if abs(f.macd_hist_norm) > 0.05:
            mom_score += 8.0
            if f.macd_hist_norm > 0:
                bullish_bias_pts += 1
            else:
                bearish_bias_pts += 1

        if 20.0 <= f.stoch_k <= 80.0:
            mom_score += 7.0
        else:
            mom_score += 4.0

        # ── 3. Volume & Flow Factor (Max 25 pts) ─────────────────────
        vol_score = 0.0
        if f.rvol_20 >= 2.0:
            vol_score += 12.0
            reasons.append(f"Heavy institutional volume (RVOL {f.rvol_20:.2f}x)")
        elif f.rvol_20 >= 1.3:
            vol_score += 7.0

        if f.cmf_20 >= 0.12:
            vol_score += 8.0
            bullish_bias_pts += 2
            reasons.append(f"Institutional accumulation (CMF +{f.cmf_20:.2f})")
        elif f.cmf_20 <= -0.12:
            vol_score += 8.0
            bearish_bias_pts += 2
            reasons.append(f"Institutional distribution (CMF {f.cmf_20:.2f})")

        # Proximity to VWAP
        if abs(f.vwap_dist_pct) <= 1.0:
            vol_score += 5.0

        # ── 4. Pattern & Catalyst Factor (Max 25 pts) ────────────────
        pattern_score = 0.0
        if f.is_bb_squeeze:
            pattern_score += 10.0
            reasons.append("Volatility Squeeze ready to expand")

        if f.has_fvg:
            pattern_score += 8.0
            reasons.append("Fair Value Gap (liquidity void) present")

        if f.bullish_pattern_count > 0:
            pattern_score += 7.0
            bullish_bias_pts += 2
            reasons.append(f"Bullish pattern: {f.strongest_pattern}")
        elif f.bearish_pattern_count > 0:
            pattern_score += 7.0
            bearish_bias_pts += 2
            reasons.append(f"Bearish pattern: {f.strongest_pattern}")

        if f.is_pinbar:
            pattern_score = min(25.0, pattern_score + 5.0)

        # ── Determine Overall Bias & Setup Category ──────────────────
        if bullish_bias_pts > bearish_bias_pts + 1:
            direction = "bullish"
        elif bearish_bias_pts > bullish_bias_pts + 1:
            direction = "bearish"
        else:
            direction = "neutral"

        # Categorize setup type
        if f.is_bb_squeeze and f.rvol_20 >= 1.4:
            setup_type = "VOLATILITY_BREAKOUT"
        elif f.mtf_is_pullback or (f.rsi_14 < 32.0 and direction == "bullish"):
            setup_type = "MEAN_REVERSION_PULLBACK"
        elif f.has_fvg:
            setup_type = "LIQUIDITY_IMBALANCE_FVG"
        else:
            setup_type = "TREND_CONTINUATION"

        composite = min(100.0, trend_score + mom_score + vol_score + pattern_score)

        return ScoredOpportunity(
            symbol=f.symbol,
            composite_score=round(composite, 1),
            direction=direction,
            setup_type=setup_type,
            factor_scores={
                "trend": trend_score,
                "momentum": mom_score,
                "volume": vol_score,
                "pattern": pattern_score,
            },
            reasons=reasons,
            features=f,
        )
