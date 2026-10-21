"""10 Super-Agents Committee — News-Aware & Regime-Adaptive Multi-Agent System."""
from __future__ import annotations

from typing import Any, Callable

from strategies.base import SignalDirection, Strategy, StrategySignal


class FeatureRuleAgent(Strategy):
    """Explainable, news-aware feature-family agent used by the 10-agent committee."""

    def __init__(
        self,
        name: str,
        description: str,
        timeframes: list[str],
        rule: Callable[[dict[str, Any]], tuple[SignalDirection, float, list[str]]],
    ) -> None:
        self._name = name
        self._description = description
        self._timeframes = timeframes
        self._rule = rule

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def timeframes(self) -> list[str]:
        return self._timeframes

    async def analyze(
        self,
        symbol: str,
        features: dict[str, Any],
        market_regime: str | None = None,
    ) -> StrategySignal | None:
        direction, score, reasons = self._rule(features)
        if direction == SignalDirection.HOLD or score < 0.55:
            return None
        price = float(features.get("close_price", 0.0))
        if price <= 0:
            return None

        news_score = float(features.get("news_composite_score", 0.0))
        is_buy = direction == SignalDirection.BUY
        if is_buy and news_score >= 25.0:
            score = min(0.98, score + min(0.12, news_score / 500.0))
            reasons.append(f"Bullish news/Gemini catalyst alignment (score {news_score:+.1f})")
        elif not is_buy and news_score <= -25.0:
            score = min(0.98, score + min(0.12, abs(news_score) / 500.0))
            reasons.append(f"Bearish news/Gemini catalyst alignment (score {news_score:+.1f})")

        atr_pct = max(0.2, float(features.get("natr_14", 1.5))) / 100.0
        risk_distance = price * min(0.08, max(0.01, atr_pct * 1.8))
        reward_mult = 2.2 if abs(news_score) >= 35.0 else 2.0

        return StrategySignal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            score=round(min(1.0, score), 3),
            confidence=round(min(1.0, score), 3),
            expected_return=round((risk_distance * reward_mult / price) * 100.0, 2),
            risk_estimate=round((risk_distance / price) * 100.0, 2),
            timeframe=features.get("timeframe", "1h"),
            stop_suggestion=round(price - risk_distance if is_buy else price + risk_distance, 2),
            target_suggestion=round(
                price + risk_distance * reward_mult if is_buy else price - risk_distance * reward_mult,
                2,
            ),
            market_regime=market_regime,
            reasons=reasons,
            metadata={
                "agent_type": self.name,
                "feature_family": self._description,
                "news_composite_score": news_score,
            },
        )

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "timeframes": self.timeframes}


def _momentum_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    rsi = float(f.get("rsi_14", 50.0))
    macd = float(f.get("macd_hist_norm", 0.0))
    roc = float(f.get("roc_10", 0.0))
    news_score = float(f.get("news_composite_score", 0.0))
    if 50 <= rsi <= 72 and (macd > 0.03 or roc > 0.8):
        return SignalDirection.BUY, 0.74 + min(0.12, max(0.0, macd) * 0.4), [f"Momentum aligned BUY: RSI {rsi:.1f}, MACD {macd:+.3f}"]
    if 28 <= rsi <= 50 and (macd < -0.03 or roc < -0.8):
        return SignalDirection.SELL, 0.74 + min(0.12, abs(min(0.0, macd)) * 0.4), [f"Momentum aligned SELL: RSI {rsi:.1f}, MACD {macd:+.3f}"]
    if news_score >= 35.0 and rsi >= 48.0 and macd >= 0.0:
        return SignalDirection.BUY, 0.76, [f"Catalyst momentum ignition BUY: News {news_score:+.1f}"]
    if news_score <= -35.0 and rsi <= 52.0 and macd <= 0.0:
        return SignalDirection.SELL, 0.76, [f"Catalyst momentum breakdown SELL: News {news_score:+.1f}"]
    return SignalDirection.HOLD, 0.0, []


def _volume_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    """Agent #7: Institutional Volume & Money Flow (RVOL + CMF + MFI + News Buzz)."""
    rvol = float(f.get("rvol_20", 0.0))
    cmf = float(f.get("cmf_20", 0.0))
    mfi = float(f.get("mfi_14", 50.0))
    buzz = float(f.get("news_buzz_ratio", 1.0))
    news_score = float(f.get("news_composite_score", 0.0))

    if rvol >= 1.25 and (cmf >= 0.08 or (mfi >= 55.0 and news_score >= 25.0)):
        score = min(0.92, 0.68 + max(0.05, cmf) + (0.04 if buzz >= 2.0 else 0.0))
        return SignalDirection.BUY, score, [f"Volume-flow accumulation: RVOL {rvol:.2f}x, CMF {cmf:.2f}"]
    if rvol >= 1.25 and (cmf <= -0.08 or (mfi <= 45.0 and news_score <= -25.0)):
        score = min(0.92, 0.68 + max(0.05, abs(cmf)) + (0.04 if buzz >= 2.0 else 0.0))
        return SignalDirection.SELL, score, [f"Volume-flow distribution: RVOL {rvol:.2f}x, CMF {cmf:.2f}"]
    return SignalDirection.HOLD, 0.0, []


def _regime_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    adx = float(f.get("adx_14", 0.0))
    supertrend = int(f.get("supertrend_dir", 0))
    ema_spread = float(f.get("ema_9_21_spread", 0.0))
    news_score = float(f.get("news_composite_score", 0.0))
    if adx >= 18 and supertrend == 1 and news_score > -40.0:
        return SignalDirection.BUY, min(0.90, 0.68 + adx / 200.0 + (0.04 if ema_spread > 0 else 0.0)), [f"Trend regime bullish: ADX {adx:.1f}"]
    if adx >= 18 and supertrend == -1 and news_score < 40.0:
        return SignalDirection.SELL, min(0.90, 0.68 + adx / 200.0 + (0.04 if ema_spread < 0 else 0.0)), [f"Trend regime bearish: ADX {adx:.1f}"]
    return SignalDirection.HOLD, 0.0, []


def _vwap_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    dist = float(f.get("vwap_dist_pct", 0.0))
    rsi = float(f.get("rsi_14", 50.0))
    adx = float(f.get("adx_14", 0.0))
    supertrend = int(f.get("supertrend_dir", 0))
    news_score = float(f.get("news_composite_score", 0.0))
    bull_cats = float(f.get("news_bullish_catalyst_count", 0.0))
    bear_cats = float(f.get("news_bearish_catalyst_count", 0.0))

    # 1. Strong News Catalyst Ignition (trades WITH the catalyst)
    if (news_score >= 30.0 or bull_cats >= 1.0) and rsi < 75.0 and supertrend >= 0:
        return (
            SignalDirection.BUY,
            0.78,
            [f"Catalyst/VWAP institutional BUY: News score {news_score:+.1f} ({int(bull_cats)} bullish catalysts)"],
        )
    if (news_score <= -30.0 or bear_cats >= 1.0) and rsi > 25.0 and supertrend <= 0:
        return (
            SignalDirection.SELL,
            0.78,
            [f"Catalyst/VWAP institutional SELL: News score {news_score:+.1f} ({int(bear_cats)} bearish catalysts)"],
        )

    # 2. Pure VWAP Reversion (only when NOT fighting a strong runaway trend)
    is_strong_uptrend = supertrend == 1 and adx >= 25.0
    is_strong_downtrend = supertrend == -1 and adx >= 25.0
    if dist <= -1.2 and rsi <= 40 and not is_strong_downtrend and news_score > -25.0:
        return SignalDirection.BUY, 0.70, [f"VWAP discount reversion BUY: distance {dist:.2f}%"]
    if dist >= 1.2 and rsi >= 60 and not is_strong_uptrend and news_score < 25.0:
        return SignalDirection.SELL, 0.70, [f"VWAP premium reversion SELL: distance {dist:.2f}%"]
    return SignalDirection.HOLD, 0.0, []


def _volatility_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    squeeze = bool(f.get("is_bb_squeeze", False))
    has_fvg = bool(f.get("has_fvg", False))
    rvol = float(f.get("rvol_20", 0.0))
    direction = int(f.get("supertrend_dir", 0))
    close_pos = float(f.get("close_position_in_bar", 0.5))
    news_score = float(f.get("news_composite_score", 0.0))

    if (squeeze or has_fvg or abs(news_score) >= 30.0) and rvol >= 1.30:
        if direction == 1 or (direction == 0 and close_pos >= 0.65 and news_score >= 20.0):
            return (
                SignalDirection.BUY,
                0.75,
                [f"Volatility expansion BUY: RVOL {rvol:.2f}x (squeeze={squeeze}, fvg={has_fvg})"],
            )
        if direction == -1 or (direction == 0 and close_pos <= 0.35 and news_score <= -20.0):
            return (
                SignalDirection.SELL,
                0.75,
                [f"Volatility breakdown SELL: RVOL {rvol:.2f}x (squeeze={squeeze}, fvg={has_fvg})"],
            )
    return SignalDirection.HOLD, 0.0, []


def build_additional_agents() -> list[Strategy]:
    return [
        FeatureRuleAgent("momentum_confirmation", "RSI, MACD, ROC and News momentum", ["1m", "5m", "1h"], _momentum_rule),
        FeatureRuleAgent("volume_flow", "RVOL, CMF and MFI institutional flow", ["1m", "5m", "1h"], _volume_rule),
        FeatureRuleAgent("regime_classifier", "ADX, EMA Ribbon and Supertrend regime", ["5m", "15m", "1d"], _regime_rule),
        FeatureRuleAgent("vwap_reversion_guard", "VWAP reversion and Gemini/News catalyst guard", ["1m", "5m", "1h"], _vwap_rule),
        FeatureRuleAgent("volatility_breakout", "Bollinger squeeze, FVG and volume expansion", ["5m", "15m", "1h"], _volatility_rule),
    ]


def build_committee_agents(existing: list[Strategy]) -> list[Strategy]:
    return (existing + build_additional_agents())[:10]


__all__ = ["FeatureRuleAgent", "build_additional_agents", "build_committee_agents"]
