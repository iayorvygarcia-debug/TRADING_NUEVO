"""
AI Quant Trading Platform — Candlestick Pattern & Price Action Recognition.

Detects single-bar, two-bar, and three-bar candlestick patterns, as well as
price action structures (Fair Value Gaps, Pinbars, Inside Bars).

Each pattern returns:
- Name
- Sentiment: Bullish (+1), Bearish (-1), or Neutral (0)
- Reliability/Strength: 0.0 to 1.0
- Bar index and descriptive metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd


class PatternSentiment(IntEnum):
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


@dataclass(slots=True)
class PatternResult:
    """Detected candlestick or price action pattern."""
    pattern_name: str
    sentiment: PatternSentiment
    strength: float  # 0.0 to 1.0
    bar_index: int
    price: float
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_bullish(self) -> bool:
        return self.sentiment == PatternSentiment.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.sentiment == PatternSentiment.BEARISH


class CandlePatternDetector:
    """
    High-precision candlestick pattern detector.
    
    Uses mathematical ratios on candle body, upper shadow, and lower shadow
    relative to total candle range.
    """

    DOJI_THRESHOLD = 0.10          # Body is <= 10% of range
    HAMMER_BODY_MAX = 0.35         # Body is <= 35% of range
    HAMMER_SHADOW_MIN = 2.0        # Lower shadow >= 2x body
    MARUBOZU_BODY_MIN = 0.85       # Body is >= 85% of range
    PINBAR_SHADOW_MIN = 0.65       # One shadow is >= 65% of entire range

    @classmethod
    def analyze_latest(cls, df: pd.DataFrame) -> list[PatternResult]:
        """
        Analyze the most recent bars in the DataFrame for all supported patterns.
        Requires at least 3 bars for 3-bar patterns.
        """
        patterns: list[PatternResult] = []
        n = len(df)
        if n < 3:
            return patterns

        idx = n - 1
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        # ── Single-bar checks on current bar ─────────────────────────
        patterns.extend(cls.detect_single_bar(curr, idx))

        # ── Two-bar checks on prev -> curr ───────────────────────────
        patterns.extend(cls.detect_two_bar(prev, curr, idx))

        # ── Three-bar checks on prev2 -> prev -> curr ────────────────
        patterns.extend(cls.detect_three_bar(prev2, prev, curr, idx))

        # ── Fair Value Gap (FVG) on prev2 -> prev -> curr ────────────
        fvg = cls.detect_fvg(prev2, prev, curr, idx)
        if fvg:
            patterns.append(fvg)

        return patterns

    @classmethod
    def scan_all(cls, df: pd.DataFrame) -> list[PatternResult]:
        """Scan the entire history for all occurrences of patterns."""
        results: list[PatternResult] = []
        n = len(df)
        if n < 3:
            return results

        for i in range(2, n):
            prev2 = df.iloc[i - 2]
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            results.extend(cls.detect_single_bar(curr, i))
            results.extend(cls.detect_two_bar(prev, curr, i))
            results.extend(cls.detect_three_bar(prev2, prev, curr, i))

            fvg = cls.detect_fvg(prev2, prev, curr, i)
            if fvg:
                results.append(fvg)

        return results

    # ── Single-Bar Detection ─────────────────────────────────────────

    @classmethod
    def detect_single_bar(cls, bar: pd.Series, bar_idx: int) -> list[PatternResult]:
        """Detect single-candle patterns."""
        patterns: list[PatternResult] = []

        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        rng = h - l
        if rng <= 1e-8:
            return patterns

        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        is_bullish_candle = c > o

        # 1. Doji
        if (body / rng) <= cls.DOJI_THRESHOLD:
            # Check dragonfly vs gravestone vs regular doji
            if lower_shadow >= 0.7 * rng:
                patterns.append(PatternResult(
                    pattern_name="dragonfly_doji",
                    sentiment=PatternSentiment.BULLISH,
                    strength=0.75,
                    bar_index=bar_idx,
                    price=c,
                    description="Dragonfly Doji: strong bullish rejection of lower prices",
                ))
            elif upper_shadow >= 0.7 * rng:
                patterns.append(PatternResult(
                    pattern_name="gravestone_doji",
                    sentiment=PatternSentiment.BEARISH,
                    strength=0.75,
                    bar_index=bar_idx,
                    price=c,
                    description="Gravestone Doji: strong bearish rejection of higher prices",
                ))
            else:
                patterns.append(PatternResult(
                    pattern_name="doji",
                    sentiment=PatternSentiment.NEUTRAL,
                    strength=0.50,
                    bar_index=bar_idx,
                    price=c,
                    description="Standard Doji: market indecision",
                ))

        # 2. Hammer (Bullish) & Hanging Man (Bearish)
        # Small body at upper end, long lower shadow (>= 2x body), negligible upper shadow
        if (body / rng) <= cls.HAMMER_BODY_MAX and lower_shadow >= (2.0 * body) and upper_shadow <= (0.2 * rng):
            patterns.append(PatternResult(
                pattern_name="hammer",
                sentiment=PatternSentiment.BULLISH,
                strength=0.80,
                bar_index=bar_idx,
                price=c,
                description="Hammer: strong rejection of lower levels",
            ))

        # 3. Inverted Hammer (Bullish) & Shooting Star (Bearish)
        # Small body at lower end, long upper shadow (>= 2x body), negligible lower shadow
        if (body / rng) <= cls.HAMMER_BODY_MAX and upper_shadow >= (2.0 * body) and lower_shadow <= (0.2 * rng):
            patterns.append(PatternResult(
                pattern_name="shooting_star",
                sentiment=PatternSentiment.BEARISH,
                strength=0.80,
                bar_index=bar_idx,
                price=c,
                description="Shooting Star: sellers rejected highs",
            ))

        # 4. Pinbar (Bullish / Bearish price action rejection)
        if lower_shadow / rng >= cls.PINBAR_SHADOW_MIN and (body / rng) <= 0.30:
            patterns.append(PatternResult(
                pattern_name="bullish_pinbar",
                sentiment=PatternSentiment.BULLISH,
                strength=0.85,
                bar_index=bar_idx,
                price=c,
                description="Bullish Pinbar: long bottom tail rejecting key support",
            ))
        elif upper_shadow / rng >= cls.PINBAR_SHADOW_MIN and (body / rng) <= 0.30:
            patterns.append(PatternResult(
                pattern_name="bearish_pinbar",
                sentiment=PatternSentiment.BEARISH,
                strength=0.85,
                bar_index=bar_idx,
                price=c,
                description="Bearish Pinbar: long upper tail rejecting key resistance",
            ))

        # 5. Marubozu (Extreme directional momentum)
        if (body / rng) >= cls.MARUBOZU_BODY_MIN:
            sentiment = PatternSentiment.BULLISH if is_bullish_candle else PatternSentiment.BEARISH
            name = "bullish_marubozu" if is_bullish_candle else "bearish_marubozu"
            patterns.append(PatternResult(
                pattern_name=name,
                sentiment=sentiment,
                strength=0.75,
                bar_index=bar_idx,
                price=c,
                description=f"{name.replace('_', ' ').title()}: massive momentum with minimal shadows",
            ))

        return patterns

    # ── Two-Bar Detection ────────────────────────────────────────────

    @classmethod
    def detect_two_bar(cls, prev: pd.Series, curr: pd.Series, bar_idx: int) -> list[PatternResult]:
        """Detect two-candle reversal and continuation patterns."""
        patterns: list[PatternResult] = []

        po, ph, pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])
        co, ch, cl, cc = float(curr["open"]), float(curr["high"]), float(curr["low"]), float(curr["close"])

        prev_bearish = pc < po
        prev_bullish = pc > po
        curr_bullish = cc > co
        curr_bearish = cc < co

        # 1. Bullish Engulfing
        # Previous is bearish, current is bullish and completely engulfs previous body
        if prev_bearish and curr_bullish:
            if co <= pc and cc >= po:
                patterns.append(PatternResult(
                    pattern_name="bullish_engulfing",
                    sentiment=PatternSentiment.BULLISH,
                    strength=0.85,
                    bar_index=bar_idx,
                    price=cc,
                    description="Bullish Engulfing: buyers overwhelmed previous selling pressure",
                ))

        # 2. Bearish Engulfing
        # Previous is bullish, current is bearish and completely engulfs previous body
        if prev_bullish and curr_bearish:
            if co >= pc and cc <= po:
                patterns.append(PatternResult(
                    pattern_name="bearish_engulfing",
                    sentiment=PatternSentiment.BEARISH,
                    strength=0.85,
                    bar_index=bar_idx,
                    price=cc,
                    description="Bearish Engulfing: sellers overwhelmed previous buying pressure",
                ))

        # 3. Piercing Line (Bullish)
        # Prev is bearish; curr opens lower but closes above 50% of prev body
        if prev_bearish and curr_bullish and co < pl and cc > (po + pc) / 2.0 and cc < po:
            patterns.append(PatternResult(
                pattern_name="piercing_line",
                sentiment=PatternSentiment.BULLISH,
                strength=0.75,
                bar_index=bar_idx,
                price=cc,
                description="Piercing Line: strong bullish counter-attack past midpoint",
            ))

        # 4. Dark Cloud Cover (Bearish)
        # Prev is bullish; curr opens higher but closes below 50% of prev body
        if prev_bullish and curr_bearish and co > ph and cc < (po + pc) / 2.0 and cc > po:
            patterns.append(PatternResult(
                pattern_name="dark_cloud_cover",
                sentiment=PatternSentiment.BEARISH,
                strength=0.75,
                bar_index=bar_idx,
                price=cc,
                description="Dark Cloud Cover: sellers push price below midpoint of prior bull bar",
            ))

        # 5. Inside Bar (Volatility compression)
        # Current bar's entire range is completely inside previous bar's range
        if ch < ph and cl > pl:
            patterns.append(PatternResult(
                pattern_name="inside_bar",
                sentiment=PatternSentiment.NEUTRAL,
                strength=0.60,
                bar_index=bar_idx,
                price=cc,
                description="Inside Bar: consolidation / pre-breakout compression",
            ))

        # 6. Tweezer Bottom (Bullish) / Tweezer Top (Bearish)
        # Matching lows/highs with opposite directions
        if abs(cl - pl) <= (0.05 * (ph - pl)):
            if prev_bearish and curr_bullish:
                patterns.append(PatternResult(
                    pattern_name="tweezer_bottom",
                    sentiment=PatternSentiment.BULLISH,
                    strength=0.70,
                    bar_index=bar_idx,
                    price=cc,
                    description="Tweezer Bottom: exact dual rejection of same support level",
                ))
        if abs(ch - ph) <= (0.05 * (ph - pl)):
            if prev_bullish and curr_bearish:
                patterns.append(PatternResult(
                    pattern_name="tweezer_top",
                    sentiment=PatternSentiment.BEARISH,
                    strength=0.70,
                    bar_index=bar_idx,
                    price=cc,
                    description="Tweezer Top: exact dual rejection of same resistance level",
                ))

        return patterns

    # ── Three-Bar Detection ──────────────────────────────────────────

    @classmethod
    def detect_three_bar(
        cls,
        prev2: pd.Series,
        prev: pd.Series,
        curr: pd.Series,
        bar_idx: int,
    ) -> list[PatternResult]:
        """Detect three-candle classic reversal patterns."""
        patterns: list[PatternResult] = []

        o1, c1 = float(prev2["open"]), float(prev2["close"])
        o2, c2 = float(prev["open"]), float(prev["close"])
        o3, c3 = float(curr["open"]), float(curr["close"])

        # 1. Morning Star (Bullish)
        # Bar 1: Strong bearish bar
        # Bar 2: Small body star (gapped down or compressed)
        # Bar 3: Strong bullish bar closing above midpoint of Bar 1
        if c1 < o1 and abs(c2 - o2) < abs(c1 - o1) * 0.4 and c3 > o3:
            if c3 > (o1 + c1) / 2.0:
                patterns.append(PatternResult(
                    pattern_name="morning_star",
                    sentiment=PatternSentiment.BULLISH,
                    strength=0.90,
                    bar_index=bar_idx,
                    price=c3,
                    description="Morning Star: premier high-probability 3-bar bottom reversal",
                ))

        # 2. Evening Star (Bearish)
        # Bar 1: Strong bullish bar
        # Bar 2: Small body star
        # Bar 3: Strong bearish bar closing below midpoint of Bar 1
        if c1 > o1 and abs(c2 - o2) < abs(c1 - o1) * 0.4 and c3 < o3:
            if c3 < (o1 + c1) / 2.0:
                patterns.append(PatternResult(
                    pattern_name="evening_star",
                    sentiment=PatternSentiment.BEARISH,
                    strength=0.90,
                    bar_index=bar_idx,
                    price=c3,
                    description="Evening Star: premier high-probability 3-bar top reversal",
                ))

        # 3. Three White Soldiers (Bullish continuation/reversal)
        if c1 > o1 and c2 > o2 and c3 > o3:
            if c3 > c2 > c1 and o3 > o2 > o1:
                patterns.append(PatternResult(
                    pattern_name="three_white_soldiers",
                    sentiment=PatternSentiment.BULLISH,
                    strength=0.85,
                    bar_index=bar_idx,
                    price=c3,
                    description="Three White Soldiers: three consecutive strong expanding bull bars",
                ))

        # 4. Three Black Crows (Bearish continuation/reversal)
        if c1 < o1 and c2 < o2 and c3 < o3:
            if c3 < c2 < c1 and o3 < o2 < o1:
                patterns.append(PatternResult(
                    pattern_name="three_black_crows",
                    sentiment=PatternSentiment.BEARISH,
                    strength=0.85,
                    bar_index=bar_idx,
                    price=c3,
                    description="Three Black Crows: three consecutive strong expanding bear bars",
                ))

        return patterns

    # ── Price Action Structure: Fair Value Gap (FVG) ────────────────

    @classmethod
    def detect_fvg(
        cls,
        bar1: pd.Series,
        bar2: pd.Series,
        bar3: pd.Series,
        bar_idx: int,
    ) -> PatternResult | None:
        """
        Detect Fair Value Gap (Imbalance / Liquidity Void).
        
        Bullish FVG: Bar 1 High < Bar 3 Low (Gap between Bar 1 High and Bar 3 Low)
        Bearish FVG: Bar 1 Low > Bar 3 High (Gap between Bar 1 Low and Bar 3 High)
        """
        h1 = float(bar1["high"])
        l1 = float(bar1["low"])
        h3 = float(bar3["high"])
        l3 = float(bar3["low"])
        c3 = float(bar3["close"])

        # Bullish FVG
        if l3 > h1:
            gap_size = l3 - h1
            return PatternResult(
                pattern_name="bullish_fvg",
                sentiment=PatternSentiment.BULLISH,
                strength=0.80,
                bar_index=bar_idx,
                price=c3,
                description=f"Bullish Fair Value Gap: unmitigated demand imbalance ({gap_size:.2f} pts)",
                metadata={"top": l3, "bottom": h1, "gap_size": gap_size},
            )

        # Bearish FVG
        if h3 < l1:
            gap_size = l1 - h3
            return PatternResult(
                pattern_name="bearish_fvg",
                sentiment=PatternSentiment.BEARISH,
                strength=0.80,
                bar_index=bar_idx,
                price=c3,
                description=f"Bearish Fair Value Gap: unmitigated supply imbalance ({gap_size:.2f} pts)",
                metadata={"top": l1, "bottom": h3, "gap_size": gap_size},
            )

        return None
