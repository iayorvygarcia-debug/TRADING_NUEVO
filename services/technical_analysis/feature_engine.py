"""
AI Quant Trading Platform — Quantitative Feature Engine.

Extracts normalized, stationary, and ML-ready features from market data bars.
Guarantees:
- Zero lookahead bias (strictly causal, t <= T)
- Produces both dictionary / dataclass format for strategy rules
  and flat NumPy vectors for ML inference (XGBoost, LightGBM)
- Handles normalization, scaling, and missing values safely
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from services.technical_analysis.indicators import (
    TechnicalIndicators,
    calculate_all_indicators,
)
from services.technical_analysis.patterns import (
    CandlePatternDetector,
    PatternResult,
    PatternSentiment,
)
from services.technical_analysis.multi_timeframe import (
    TimeframeAlignment,
    TrendDirection,
)


@dataclass(slots=True)
class FeatureVector:
    """Standardized quantitative feature representation for a single bar."""
    # Metadata
    symbol: str
    timestamp: datetime
    timeframe: str
    close_price: float

    # Returns & Price Ratios (Stationary)
    return_1: float
    return_5: float
    return_20: float
    high_low_ratio: float
    close_position_in_bar: float  # (close - low) / (high - low)

    # Trend
    ema_9_dist_pct: float         # (close - EMA9) / EMA9 * 100
    ema_21_dist_pct: float        # (close - EMA21) / EMA21 * 100
    ema_50_dist_pct: float        # (close - EMA50) / EMA50 * 100
    sma_200_dist_pct: float       # (close - SMA200) / SMA200 * 100
    ema_9_21_spread: float        # (EMA9 - EMA21) / close * 100
    supertrend_dir: int           # +1, -1, or 0
    adx_14: float
    di_spread: float              # plus_di - minus_di

    # Momentum
    rsi_14: float
    macd_hist_norm: float         # macd_hist / close * 100
    stoch_k: float
    stoch_d: float
    stoch_rsi_k: float
    roc_10: float
    cci_20: float
    williams_r: float

    # Volatility
    natr_14: float                # ATR / close * 100
    bb_percent: float             # %B [0 to 1 normally]
    bb_bandwidth: float           # (upper - lower) / middle
    is_bb_squeeze: bool           # Bollinger Bands inside Keltner Channel (consolidation breakout precursor)
    hist_vol_20: float

    # Volume & Money Flow
    rvol_20: float                # Relative volume vs 20-period avg
    cmf_20: float                 # Chaikin Money Flow [-1.0, +1.0]
    mfi_14: float                 # Money Flow Index [0, 100]
    vwap_dist_pct: float          # (close - VWAP) / VWAP * 100

    # Patterns & Price Action
    bullish_pattern_count: int
    bearish_pattern_count: int
    strongest_pattern: str
    pattern_sentiment_score: float  # Sum of pattern sentiments * strengths
    has_fvg: bool
    is_pinbar: bool
    is_inside_bar: bool

    # Multi-Timeframe Alignment (Optional / Defaults)
    mtf_alignment_score: float = 0.0
    mtf_is_pullback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard Python dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_ml_features(self) -> np.ndarray:
        """
        Extract numerical features as a 1D float32 NumPy array for ML models.
        Ignores non-numeric metadata.
        """
        return np.array([
            self.return_1,
            self.return_5,
            self.return_20,
            self.high_low_ratio,
            self.close_position_in_bar,
            self.ema_9_dist_pct,
            self.ema_21_dist_pct,
            self.ema_50_dist_pct,
            self.sma_200_dist_pct,
            self.ema_9_21_spread,
            float(self.supertrend_dir),
            self.adx_14,
            self.di_spread,
            self.rsi_14,
            self.macd_hist_norm,
            self.stoch_k,
            self.stoch_d,
            self.stoch_rsi_k,
            self.roc_10,
            self.cci_20,
            self.williams_r,
            self.natr_14,
            self.bb_percent,
            self.bb_bandwidth,
            1.0 if self.is_bb_squeeze else 0.0,
            self.hist_vol_20,
            self.rvol_20,
            self.cmf_20,
            self.mfi_14,
            self.vwap_dist_pct,
            float(self.bullish_pattern_count),
            float(self.bearish_pattern_count),
            self.pattern_sentiment_score,
            1.0 if self.has_fvg else 0.0,
            1.0 if self.is_pinbar else 0.0,
            1.0 if self.is_inside_bar else 0.0,
            self.mtf_alignment_score,
            1.0 if self.mtf_is_pullback else 0.0,
        ], dtype=np.float32)

    @classmethod
    def feature_names(cls) -> list[str]:
        """Names of the features in order returned by to_ml_features()."""
        return [
            "return_1", "return_5", "return_20", "high_low_ratio", "close_position_in_bar",
            "ema_9_dist_pct", "ema_21_dist_pct", "ema_50_dist_pct", "sma_200_dist_pct",
            "ema_9_21_spread", "supertrend_dir", "adx_14", "di_spread",
            "rsi_14", "macd_hist_norm", "stoch_k", "stoch_d", "stoch_rsi_k",
            "roc_10", "cci_20", "williams_r",
            "natr_14", "bb_percent", "bb_bandwidth", "is_bb_squeeze", "hist_vol_20",
            "rvol_20", "cmf_20", "mfi_14", "vwap_dist_pct",
            "bullish_pattern_count", "bearish_pattern_count", "pattern_sentiment_score",
            "has_fvg", "is_pinbar", "is_inside_bar",
            "mtf_alignment_score", "mtf_is_pullback",
        ]


class FeatureEngine:
    """
    Quantitative Feature Generator.
    
    Transforms OHLCV historical bars into structured FeatureVector instances.
    """

    MIN_BARS_REQUIRED = 30

    @classmethod
    def compute_features(
        cls,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1d",
        mtf_alignment: TimeframeAlignment | None = None,
    ) -> FeatureVector:
        """
        Compute complete feature set for the latest closed bar of the given DataFrame.
        """
        if len(df) < cls.MIN_BARS_REQUIRED:
            raise ValueError(
                f"FeatureEngine requires at least {cls.MIN_BARS_REQUIRED} bars for robust calculations. "
                f"Got {len(df)} bars."
            )

        # 1. Compute full suite of indicators
        ind_df = calculate_all_indicators(df)
        last_row = ind_df.iloc[-1]
        c = float(last_row["close"])
        h = float(last_row["high"])
        l = float(last_row["low"])
        v = float(last_row["volume"])
        rng = max(1e-8, h - l)

        # 2. Extract Timestamp
        if isinstance(last_row.name, (pd.Timestamp, datetime)):
            ts = pd.to_datetime(last_row.name).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        elif "time" in last_row:
            ts = pd.to_datetime(last_row["time"]).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        # 3. Returns
        ret_1 = float((c - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100) if len(df) >= 2 else 0.0
        ret_5 = float((c - df["close"].iloc[-6]) / df["close"].iloc[-6] * 100) if len(df) >= 6 else 0.0
        ret_20 = float((c - df["close"].iloc[-21]) / df["close"].iloc[-21] * 100) if len(df) >= 21 else 0.0

        # 4. Moving Average Distances
        ema_9 = float(last_row.get("ema_9", c))
        ema_21 = float(last_row.get("ema_21", c))
        ema_50 = float(last_row.get("sma_50", c))
        sma_200 = float(last_row.get("sma_200", c))

        ema_9_dist = ((c - ema_9) / ema_9 * 100) if not np.isnan(ema_9) else 0.0
        ema_21_dist = ((c - ema_21) / ema_21 * 100) if not np.isnan(ema_21) else 0.0
        ema_50_dist = ((c - ema_50) / ema_50 * 100) if not np.isnan(ema_50) else 0.0
        sma_200_dist = ((c - sma_200) / sma_200 * 100) if not np.isnan(sma_200) else 0.0
        ema_spread = ((ema_9 - ema_21) / c * 100) if not (np.isnan(ema_9) or np.isnan(ema_21)) else 0.0

        # 5. Volatility & Squeeze Detection
        # Check if Bollinger Bands are inside Keltner Channel (TTM Squeeze definition)
        bb_upper = float(last_row.get("bb_upper", 0.0))
        bb_lower = float(last_row.get("bb_lower", 0.0))
        kc_upper, _, kc_lower = TechnicalIndicators.keltner_channels(df["high"], df["low"], df["close"])
        kc_u = float(kc_upper.iloc[-1]) if not kc_upper.empty else 0.0
        kc_l = float(kc_lower.iloc[-1]) if not kc_lower.empty else 0.0
        is_squeeze = (bb_lower > kc_l) and (bb_upper < kc_u) if (bb_lower and kc_l) else False

        # 6. Candlestick & Price Action Patterns
        patterns: list[PatternResult] = CandlePatternDetector.analyze_latest(df)
        bullish_p = [p for p in patterns if p.sentiment == PatternSentiment.BULLISH]
        bearish_p = [p for p in patterns if p.sentiment == PatternSentiment.BEARISH]
        sentiment_score = sum(int(p.sentiment) * p.strength for p in patterns)

        strongest = ""
        if patterns:
            strongest_p = max(patterns, key=lambda x: x.strength)
            strongest = strongest_p.pattern_name

        has_fvg = any("fvg" in p.pattern_name for p in patterns)
        is_pinbar = any("pinbar" in p.pattern_name for p in patterns)
        is_inside_bar = any("inside_bar" in p.pattern_name for p in patterns)

        # 7. Multi-Timeframe parameters
        mtf_score = mtf_alignment.alignment_score if mtf_alignment else 0.0
        mtf_pullback = mtf_alignment.is_pullback if mtf_alignment else False

        # 8. Safe values / NaN fallback
        def _safe(val: Any, default: float = 0.0) -> float:
            if val is None or np.isnan(val) or np.isinf(val):
                return default
            return float(val)

        return FeatureVector(
            symbol=symbol,
            timestamp=ts,
            timeframe=timeframe,
            close_price=c,
            return_1=round(ret_1, 4),
            return_5=round(ret_5, 4),
            return_20=round(ret_20, 4),
            high_low_ratio=round((h / max(1e-8, l)) - 1.0, 4),
            close_position_in_bar=round((c - l) / rng, 4),
            ema_9_dist_pct=round(ema_9_dist, 4),
            ema_21_dist_pct=round(ema_21_dist, 4),
            ema_50_dist_pct=round(ema_50_dist, 4),
            sma_200_dist_pct=round(sma_200_dist, 4),
            ema_9_21_spread=round(ema_spread, 4),
            supertrend_dir=int(_safe(last_row.get("supertrend_dir"), 0)),
            adx_14=round(_safe(last_row.get("adx"), 0.0), 2),
            di_spread=round(_safe(last_row.get("plus_di")) - _safe(last_row.get("minus_di")), 2),
            rsi_14=round(_safe(last_row.get("rsi_14"), 50.0), 2),
            macd_hist_norm=round((_safe(last_row.get("macd_hist")) / c) * 100, 4),
            stoch_k=round(_safe(last_row.get("stoch_k"), 50.0), 2),
            stoch_d=round(_safe(last_row.get("stoch_d"), 50.0), 2),
            stoch_rsi_k=round(_safe(last_row.get("stoch_rsi_k"), 50.0), 2),
            roc_10=round(_safe(last_row.get("roc_10"), 0.0), 4),
            cci_20=round(_safe(last_row.get("cci_20"), 0.0), 2),
            williams_r=round(_safe(last_row.get("williams_r"), -50.0), 2),
            natr_14=round(_safe(last_row.get("natr_14"), 0.0), 4),
            bb_percent=round(_safe(last_row.get("bb_percent"), 0.5), 4),
            bb_bandwidth=round(_safe(last_row.get("bb_bandwidth"), 0.0), 4),
            is_bb_squeeze=is_squeeze,
            hist_vol_20=round(_safe(last_row.get("hist_vol_20"), 0.0), 4),
            rvol_20=round(_safe(last_row.get("rvol_20"), 1.0), 2),
            cmf_20=round(_safe(last_row.get("cmf_20"), 0.0), 4),
            mfi_14=round(_safe(last_row.get("mfi_14"), 50.0), 2),
            vwap_dist_pct=round((c - _safe(last_row.get("vwap"), c)) / _safe(last_row.get("vwap"), c) * 100, 4),
            bullish_pattern_count=len(bullish_p),
            bearish_pattern_count=len(bearish_p),
            strongest_pattern=strongest,
            pattern_sentiment_score=round(sentiment_score, 3),
            has_fvg=has_fvg,
            is_pinbar=is_pinbar,
            is_inside_bar=is_inside_bar,
            mtf_alignment_score=mtf_score,
            mtf_is_pullback=mtf_pullback,
        )
