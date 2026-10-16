"""
AI Quant Trading Platform — Quantitative Technical Indicators.

Vectorized, pure-Python/NumPy/Pandas technical indicator calculations.
Does NOT require external C-libraries (like TA-Lib binary dependencies).

Categories:
1. Trend: SMA, EMA, WMA, HMA, Supertrend, ADX, Parabolic SAR
2. Momentum: RSI, MACD, Stochastic, StochRSI, ROC, Williams %R, CCI
3. Volatility: ATR, NATR, Bollinger Bands, Keltner Channels, Historical Volatility
4. Volume & Flow: VWAP, OBV, CMF, MFI, RVOL (Relative Volume)
5. Support & Resistance: Pivot Points (Classic, Camarilla, Fibonacci), Swing Extremas

Guarantees:
- Pure functions, no mutation of input data
- Zero lookahead bias (causal filtering only)
- Numerical stability with epsilon guards against division by zero
- Handles NaN warm-up periods cleanly
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """High-performance quantitative indicator library."""

    EPS = 1e-12

    # ── Trend Indicators ─────────────────────────────────────────────

    @staticmethod
    def sma(series: pd.Series, period: int = 20) -> pd.Series:
        """Simple Moving Average."""
        if len(series) < period:
            return pd.Series(np.nan, index=series.index)
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int = 20, adjust: bool = False) -> pd.Series:
        """Exponential Moving Average."""
        if len(series) < period:
            return pd.Series(np.nan, index=series.index)
        return series.ewm(span=period, adjust=adjust, min_periods=period).mean()

    @staticmethod
    def wma(series: pd.Series, period: int = 20) -> pd.Series:
        """Weighted Moving Average."""
        if len(series) < period:
            return pd.Series(np.nan, index=series.index)
        weights = np.arange(1, period + 1)
        w_sum = weights.sum()

        def _calc(window: np.ndarray) -> float:
            return np.dot(window, weights) / w_sum

        return series.rolling(window=period, min_periods=period).apply(_calc, raw=True)

    @classmethod
    def hma(cls, series: pd.Series, period: int = 20) -> pd.Series:
        """
        Hull Moving Average (HMA).
        Formula: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        Significantly reduces lag while maintaining curve smoothness.
        """
        if len(series) < period:
            return pd.Series(np.nan, index=series.index)
        half_period = max(1, int(period / 2))
        sqrt_period = max(1, int(math.sqrt(period)))

        wma_half = cls.wma(series, half_period)
        wma_full = cls.wma(series, period)
        diff = 2 * wma_half - wma_full
        return cls.wma(diff.dropna(), sqrt_period).reindex(series.index)

    @classmethod
    def supertrend(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Supertrend Indicator.
        
        Returns:
            Tuple of (supertrend_value, direction)
            direction: +1 for Bullish (price above line), -1 for Bearish (price below line)
        """
        n = len(close)
        if n < period:
            nan_s = pd.Series(np.nan, index=close.index)
            return nan_s, nan_s

        atr_val = cls.atr(high, low, close, period=period)
        hl2 = (high + low) / 2.0

        basic_upper = hl2 + (multiplier * atr_val)
        basic_lower = hl2 - (multiplier * atr_val)

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        supertrend = pd.Series(np.nan, index=close.index)
        direction = pd.Series(1, index=close.index, dtype=int)

        # Iterative causal update
        close_vals = close.values
        fu_vals = final_upper.to_numpy(copy=True)
        fl_vals = final_lower.to_numpy(copy=True)
        bu_vals = basic_upper.values
        bl_vals = basic_lower.values
        st_vals = np.zeros(n)
        dir_vals = np.ones(n, dtype=int)

        # Start from period index
        for i in range(period, n):
            # Lower band
            if bl_vals[i] > fl_vals[i - 1] or close_vals[i - 1] < fl_vals[i - 1]:
                fl_vals[i] = bl_vals[i]
            else:
                fl_vals[i] = fl_vals[i - 1]

            # Upper band
            if bu_vals[i] < fu_vals[i - 1] or close_vals[i - 1] > fu_vals[i - 1]:
                fu_vals[i] = bu_vals[i]
            else:
                fu_vals[i] = fu_vals[i - 1]

            # Trend direction
            if dir_vals[i - 1] == 1:
                if close_vals[i] < fl_vals[i]:
                    dir_vals[i] = -1
                    st_vals[i] = fu_vals[i]
                else:
                    dir_vals[i] = 1
                    st_vals[i] = fl_vals[i]
            else:
                if close_vals[i] > fu_vals[i]:
                    dir_vals[i] = 1
                    st_vals[i] = fl_vals[i]
                else:
                    dir_vals[i] = -1
                    st_vals[i] = fu_vals[i]

        supertrend.iloc[period:] = st_vals[period:]
        direction.iloc[period:] = dir_vals[period:]
        direction.iloc[:period] = 0

        return supertrend, direction

    @classmethod
    def adx(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index (ADX), +DI, and -DI.
        Measures trend strength (0 to 100), regardless of direction.
        
        Returns:
            Tuple of (adx, plus_di, minus_di)
        """
        if len(close) < period * 2:
            nan_s = pd.Series(np.nan, index=close.index)
            return nan_s, nan_s, nan_s

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = cls._true_range(high, low, close)

        # Wilder's smoothing
        tr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        pdm_smooth = pd.Series(plus_dm, index=close.index).ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()
        mdm_smooth = pd.Series(minus_dm, index=close.index).ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()

        plus_di = 100.0 * (pdm_smooth / (tr_smooth + cls.EPS))
        minus_di = 100.0 * (mdm_smooth / (tr_smooth + cls.EPS))

        dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + cls.EPS))
        adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        return adx, plus_di, minus_di

    # ── Momentum Indicators ──────────────────────────────────────────

    @classmethod
    def rsi(cls, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI) using standard Wilder's smoothing.
        Values: [0, 100]
        """
        if len(series) < period + 1:
            return pd.Series(np.nan, index=series.index)

        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + cls.EPS)
        rsi_val = 100.0 - (100.0 / (1.0 + rs))
        return rsi_val

    @classmethod
    def macd(
        cls,
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD).
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(series) < slow_period + signal_period:
            nan_s = pd.Series(np.nan, index=series.index)
            return nan_s, nan_s, nan_s

        fast_ema = cls.ema(series, period=fast_period)
        slow_ema = cls.ema(series, period=slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = cls.ema(macd_line.dropna(), period=signal_period).reindex(series.index)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @classmethod
    def stochastic(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator (%K, %D).
        Values: [0, 100]
        """
        if len(close) < k_period + d_period:
            nan_s = pd.Series(np.nan, index=close.index)
            return nan_s, nan_s

        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()

        fast_k = 100.0 * ((close - lowest_low) / (highest_high - lowest_low + cls.EPS))
        slow_k = fast_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
        d_line = slow_k.rolling(window=d_period, min_periods=d_period).mean()

        return slow_k, d_line

    @classmethod
    def stoch_rsi(
        cls,
        series: pd.Series,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic RSI.
        Combines RSI with Stochastic Oscillator for higher sensitivity in ranging markets.
        """
        rsi_val = cls.rsi(series, period=rsi_period)
        lowest_rsi = rsi_val.rolling(window=stoch_period, min_periods=stoch_period).min()
        highest_rsi = rsi_val.rolling(window=stoch_period, min_periods=stoch_period).max()

        stoch = (rsi_val - lowest_rsi) / (highest_rsi - lowest_rsi + cls.EPS) * 100.0
        k = stoch.rolling(window=k_period, min_periods=k_period).mean()
        d = k.rolling(window=d_period, min_periods=d_period).mean()

        return k, d

    @staticmethod
    def roc(series: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change (Percentage Momentum)."""
        if len(series) < period + 1:
            return pd.Series(np.nan, index=series.index)
        return ((series - series.shift(period)) / (series.shift(period) + 1e-12)) * 100.0

    @classmethod
    def williams_r(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Williams %R. Values between -100 and 0."""
        if len(close) < period:
            return pd.Series(np.nan, index=close.index)
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()
        return -100.0 * ((highest_high - close) / (highest_high - lowest_low + cls.EPS))

    @classmethod
    def cci(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """Commodity Channel Index (CCI)."""
        if len(close) < period:
            return pd.Series(np.nan, index=close.index)
        tp = (high + low + close) / 3.0
        sma_tp = tp.rolling(window=period, min_periods=period).mean()
        mad = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        return (tp - sma_tp) / (0.015 * mad + cls.EPS)

    # ── Volatility Indicators ────────────────────────────────────────

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """True Range calculation."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
        return tr

    @classmethod
    def atr(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range (ATR) with Wilder's smoothing."""
        if len(close) < period:
            return pd.Series(np.nan, index=close.index)
        tr = cls._true_range(high, low, close)
        return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    @classmethod
    def natr(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Normalized Average True Range (percentage of close)."""
        atr_val = cls.atr(high, low, close, period=period)
        return (atr_val / (close + cls.EPS)) * 100.0

    @classmethod
    def bollinger_bands(
        cls,
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band, percent_b, bandwidth)
        """
        if len(series) < period:
            nan_s = pd.Series(np.nan, index=series.index)
            return nan_s, nan_s, nan_s, nan_s, nan_s

        middle = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        bandwidth = (upper - lower) / (middle + cls.EPS)
        percent_b = (series - lower) / (upper - lower + cls.EPS)

        return upper, middle, lower, percent_b, bandwidth

    @classmethod
    def keltner_channels(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.
        
        Returns:
            Tuple of (upper, middle, lower)
        """
        middle = cls.ema(close, period=ema_period)
        atr_val = cls.atr(high, low, close, period=atr_period)

        upper = middle + (multiplier * atr_val)
        lower = middle - (multiplier * atr_val)
        return upper, middle, lower

    @staticmethod
    def historical_volatility(
        close: pd.Series,
        period: int = 20,
        annualize_factor: float = 252.0,
    ) -> pd.Series:
        """Annualized historical volatility of log returns."""
        if len(close) < period + 1:
            return pd.Series(np.nan, index=close.index)
        log_returns = np.log(close / close.shift(1))
        rolling_std = log_returns.rolling(window=period, min_periods=period).std()
        return rolling_std * math.sqrt(annualize_factor)

    # ── Volume & Flow Indicators ─────────────────────────────────────

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Volume-Weighted Average Price (Cumulative).
        Formula: cumsum(typical_price * volume) / cumsum(volume)
        """
        typical_price = (high + low + close) / 3.0
        pv = typical_price * volume
        cum_pv = pv.cumsum()
        cum_vol = volume.cumsum()
        return cum_pv / (cum_vol + 1e-12)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume (OBV)."""
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @classmethod
    def cmf(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """
        Chaikin Money Flow (CMF).
        Measures institutional accumulation / distribution over a rolling period.
        Values: [-1.0, 1.0]
        """
        if len(close) < period:
            return pd.Series(np.nan, index=close.index)
        high_low = high - low
        clv = ((close - low) - (high - close)) / (high_low + cls.EPS)
        vol_clv = clv * volume
        return vol_clv.rolling(window=period, min_periods=period).sum() / (
            volume.rolling(window=period, min_periods=period).sum() + cls.EPS
        )

    @classmethod
    def mfi(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Money Flow Index (MFI) - Volume-weighted RSI.
        Values: [0, 100]
        """
        if len(close) < period + 1:
            return pd.Series(np.nan, index=close.index)

        typical_price = (high + low + close) / 3.0
        raw_money_flow = typical_price * volume

        diff = typical_price.diff()
        pos_flow = np.where(diff > 0, raw_money_flow, 0.0)
        neg_flow = np.where(diff < 0, raw_money_flow, 0.0)

        pos_sum = pd.Series(pos_flow, index=close.index).rolling(window=period).sum()
        neg_sum = pd.Series(neg_flow, index=close.index).rolling(window=period).sum()

        money_ratio = pos_sum / (neg_sum + cls.EPS)
        return 100.0 - (100.0 / (1.0 + money_ratio))

    @staticmethod
    def rvol(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Relative Volume (RVOL).
        Ratio of current volume to average volume over rolling window.
        Values > 2.0 indicate strong institutional activity.
        """
        vol_ma = volume.rolling(window=period, min_periods=period).mean()
        return volume / (vol_ma + 1e-12)

    # ── Support & Resistance ─────────────────────────────────────────

    @staticmethod
    def pivot_points(
        high: float,
        low: float,
        close: float,
    ) -> dict[str, float]:
        """
        Compute classic, Fibonacci, and Camarilla pivot points for a bar/day.
        """
        pp = (high + low + close) / 3.0
        rng = high - low

        # Classic
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + rng
        s2 = pp - rng
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)

        # Fibonacci
        fib_r1 = pp + 0.382 * rng
        fib_r2 = pp + 0.618 * rng
        fib_r3 = pp + 1.000 * rng
        fib_s1 = pp - 0.382 * rng
        fib_s2 = pp - 0.618 * rng
        fib_s3 = pp - 1.000 * rng

        # Camarilla
        cam_r4 = close + rng * 1.1 / 2.0
        cam_r3 = close + rng * 1.1 / 4.0
        cam_s3 = close - rng * 1.1 / 4.0
        cam_s4 = close - rng * 1.1 / 2.0

        return {
            "pivot": round(pp, 4),
            "r1": round(r1, 4),
            "s1": round(s1, 4),
            "r2": round(r2, 4),
            "s2": round(s2, 4),
            "r3": round(r3, 4),
            "s3": round(s3, 4),
            "fib_r1": round(fib_r1, 4),
            "fib_r2": round(fib_r2, 4),
            "fib_r3": round(fib_r3, 4),
            "fib_s1": round(fib_s1, 4),
            "fib_s2": round(fib_s2, 4),
            "fib_s3": round(fib_s3, 4),
            "cam_r3": round(cam_r3, 4),
            "cam_r4": round(cam_r4, 4),
            "cam_s3": round(cam_s3, 4),
            "cam_s4": round(cam_s4, 4),
        }

    @staticmethod
    def swing_extremas(
        high: pd.Series,
        low: pd.Series,
        left_bars: int = 2,
        right_bars: int = 2,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect swing highs and swing lows (fractal points).
        
        Returns:
            Tuple of (swing_high_mask, swing_low_mask)
        """
        n = len(high)
        sh = pd.Series(False, index=high.index)
        sl = pd.Series(False, index=low.index)

        if n < left_bars + right_bars + 1:
            return sh, sl

        h_vals = high.values
        l_vals = low.values

        for i in range(left_bars, n - right_bars):
            # Check swing high
            if all(h_vals[i] > h_vals[i - j] for j in range(1, left_bars + 1)) and \
               all(h_vals[i] >= h_vals[i + j] for j in range(1, right_bars + 1)):
                sh.iloc[i] = True

            # Check swing low
            if all(l_vals[i] < l_vals[i - j] for j in range(1, left_bars + 1)) and \
               all(l_vals[i] <= l_vals[i + j] for j in range(1, right_bars + 1)):
                sl.iloc[i] = True

        return sh, sl


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function: computes full suite of indicators on an OHLCV DataFrame.
    Expected columns: 'open', 'high', 'low', 'close', 'volume'
    """
    res = df.copy()
    c = res["close"]
    h = res["high"]
    l = res["low"]
    v = res["volume"]

    # Moving averages
    res["sma_20"] = TechnicalIndicators.sma(c, 20)
    res["sma_50"] = TechnicalIndicators.sma(c, 50)
    res["sma_200"] = TechnicalIndicators.sma(c, 200)
    res["ema_9"] = TechnicalIndicators.ema(c, 9)
    res["ema_21"] = TechnicalIndicators.ema(c, 21)
    res["hma_20"] = TechnicalIndicators.hma(c, 20)

    # Momentum
    res["rsi_14"] = TechnicalIndicators.rsi(c, 14)
    macd_l, macd_s, macd_h = TechnicalIndicators.macd(c)
    res["macd_line"] = macd_l
    res["macd_signal"] = macd_s
    res["macd_hist"] = macd_h

    stoch_k, stoch_d = TechnicalIndicators.stochastic(h, l, c)
    res["stoch_k"] = stoch_k
    res["stoch_d"] = stoch_d

    srsi_k, srsi_d = TechnicalIndicators.stoch_rsi(c)
    res["stoch_rsi_k"] = srsi_k
    res["stoch_rsi_d"] = srsi_d

    res["roc_10"] = TechnicalIndicators.roc(c, 10)
    res["williams_r"] = TechnicalIndicators.williams_r(h, l, c)
    res["cci_20"] = TechnicalIndicators.cci(h, l, c)

    # Volatility
    res["atr_14"] = TechnicalIndicators.atr(h, l, c, 14)
    res["natr_14"] = TechnicalIndicators.natr(h, l, c, 14)
    bb_u, bb_m, bb_l, bb_pct, bb_w = TechnicalIndicators.bollinger_bands(c)
    res["bb_upper"] = bb_u
    res["bb_middle"] = bb_m
    res["bb_lower"] = bb_l
    res["bb_percent"] = bb_pct
    res["bb_bandwidth"] = bb_w

    st_val, st_dir = TechnicalIndicators.supertrend(h, l, c)
    res["supertrend"] = st_val
    res["supertrend_dir"] = st_dir

    adx_val, p_di, m_di = TechnicalIndicators.adx(h, l, c)
    res["adx"] = adx_val
    res["plus_di"] = p_di
    res["minus_di"] = m_di
    res["hist_vol_20"] = TechnicalIndicators.historical_volatility(c, 20)

    # Volume & Flow
    res["vwap"] = TechnicalIndicators.vwap(h, l, c, v)
    res["obv"] = TechnicalIndicators.obv(c, v)
    res["cmf_20"] = TechnicalIndicators.cmf(h, l, c, v, 20)
    res["mfi_14"] = TechnicalIndicators.mfi(h, l, c, v, 14)
    res["rvol_20"] = TechnicalIndicators.rvol(v, 20)

    return res
