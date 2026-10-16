"""
AI Quant Trading Platform — Multi-Timeframe Analysis Engine.

Handles:
1. Resampling of base time-series (1m bars) into higher timeframes (5m, 15m, 1h, 4h, 1d).
2. Multi-Timeframe Trend & Momentum Alignment (Macro vs Execution).
3. Confluence Detection: E.g., Higher TF Trend (1D/1H Bullish) + Lower TF Pullback (15m/5m Oversold)
   which provides the highest risk-reward entry setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from services.technical_analysis.indicators import TechnicalIndicators


class TrendDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class TimeframeAlignment:
    """Summary of trend and momentum alignment across multiple timeframes."""
    macro_tf: str           # e.g., "1d"
    inter_tf: str           # e.g., "1h"
    exec_tf: str            # e.g., "5m" or "15m"
    
    macro_trend: TrendDirection
    inter_trend: TrendDirection
    exec_trend: TrendDirection
    
    alignment_score: float  # -1.0 (all strongly bearish) to +1.0 (all strongly bullish)
    is_aligned: bool        # True if macro and intermediate agree
    is_pullback: bool       # True if macro/inter bullish but exec pulled back (dip buy opportunity)
    
    macro_details: dict[str, Any]
    inter_details: dict[str, Any]
    exec_details: dict[str, Any]


class MultiTimeframeAggregator:
    """
    Resamples OHLCV bars into multiple timeframes and evaluates confluence.
    """

    TIMEFRAME_PANDAS_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }

    @classmethod
    def resample(cls, df_1m: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample 1-minute OHLCV data to a higher timeframe.
        Requires DatetimeIndex on df_1m.
        """
        if df_1m.empty:
            return pd.DataFrame()

        freq = cls.TIMEFRAME_PANDAS_MAP.get(target_timeframe)
        if not freq:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")

        df = df_1m.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df.set_index("time", inplace=True)
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or a 'time'/'timestamp' column")

        # Standard OHLCV aggregation
        resampled = df.resample(freq, closed="left", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled

    @classmethod
    def evaluate_trend(cls, df: pd.DataFrame) -> tuple[TrendDirection, dict[str, Any]]:
        """
        Evaluate trend on a specific timeframe using EMA 21, EMA 50, and Supertrend.
        """
        if len(df) < 50:
            # Not enough data for robust trend
            return TrendDirection.NEUTRAL, {"reason": "insufficient_data"}

        c = df["close"]
        h = df["high"]
        l = df["low"]

        ema_21 = TechnicalIndicators.ema(c, 21).iloc[-1]
        ema_50 = TechnicalIndicators.ema(c, 50).iloc[-1]
        rsi_14 = TechnicalIndicators.rsi(c, 14).iloc[-1]
        _, st_dir = TechnicalIndicators.supertrend(h, l, c, period=10, multiplier=3.0)
        curr_st_dir = st_dir.iloc[-1]
        curr_price = c.iloc[-1]

        bullish_votes = 0
        bearish_votes = 0

        # EMA alignment
        if curr_price > ema_21 > ema_50:
            bullish_votes += 2
        elif curr_price < ema_21 < ema_50:
            bearish_votes += 2
        elif curr_price > ema_50:
            bullish_votes += 1
        elif curr_price < ema_50:
            bearish_votes += 1

        # Supertrend
        if curr_st_dir == 1:
            bullish_votes += 2
        elif curr_st_dir == -1:
            bearish_votes += 2

        # RSI momentum
        if rsi_14 > 55:
            bullish_votes += 1
        elif rsi_14 < 45:
            bearish_votes += 1

        total_votes = bullish_votes - bearish_votes
        if total_votes >= 2:
            direction = TrendDirection.BULLISH
        elif total_votes <= -2:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL

        details = {
            "price": float(curr_price),
            "ema_21": float(ema_21),
            "ema_50": float(ema_50),
            "rsi_14": float(rsi_14),
            "supertrend_dir": int(curr_st_dir),
            "score": total_votes,
        }

        return direction, details

    @classmethod
    def analyze_alignment(
        cls,
        macro_df: pd.DataFrame,
        inter_df: pd.DataFrame,
        exec_df: pd.DataFrame,
        macro_tf: str = "1d",
        inter_tf: str = "1h",
        exec_tf: str = "15m",
    ) -> TimeframeAlignment:
        """
        Analyze multi-timeframe confluence across Macro (e.g. 1d), Intermediate (e.g. 1h),
        and Execution (e.g. 15m) timeframes.
        """
        macro_trend, macro_det = cls.evaluate_trend(macro_df)
        inter_trend, inter_det = cls.evaluate_trend(inter_df)
        exec_trend, exec_det = cls.evaluate_trend(exec_df)

        # Weights: Macro (0.50), Inter (0.30), Exec (0.20)
        weight_map = {
            TrendDirection.BULLISH: 1.0,
            TrendDirection.NEUTRAL: 0.0,
            TrendDirection.BEARISH: -1.0,
        }
        score = (
            0.50 * weight_map[macro_trend] +
            0.30 * weight_map[inter_trend] +
            0.20 * weight_map[exec_trend]
        )

        is_aligned = (
            macro_trend == inter_trend and
            macro_trend != TrendDirection.NEUTRAL
        )

        # High-probability Pullback Setup:
        # Macro and Inter are BULLISH, but Exec TF is pulling back (e.g. Exec RSI < 42 or Exec price near EMA)
        is_pullback = False
        if macro_trend == TrendDirection.BULLISH and inter_trend == TrendDirection.BULLISH:
            exec_rsi = exec_det.get("rsi_14", 50.0)
            if exec_trend in (TrendDirection.NEUTRAL, TrendDirection.BEARISH) or exec_rsi < 45.0:
                is_pullback = True
        elif macro_trend == TrendDirection.BEARISH and inter_trend == TrendDirection.BEARISH:
            exec_rsi = exec_det.get("rsi_14", 50.0)
            if exec_trend in (TrendDirection.NEUTRAL, TrendDirection.BULLISH) or exec_rsi > 55.0:
                is_pullback = True

        return TimeframeAlignment(
            macro_tf=macro_tf,
            inter_tf=inter_tf,
            exec_tf=exec_tf,
            macro_trend=macro_trend,
            inter_trend=inter_trend,
            exec_trend=exec_trend,
            alignment_score=round(score, 3),
            is_aligned=is_aligned,
            is_pullback=is_pullback,
            macro_details=macro_det,
            inter_details=inter_det,
            exec_details=exec_det,
        )
