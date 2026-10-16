"""Additional deterministic agents for the trading committee.

These agents produce signals only. They cannot submit orders or bypass risk controls.
Each agent uses a distinct feature family while sharing the same typed signal contract.
"""
from __future__ import annotations

from typing import Any, Callable

from strategies.base import SignalDirection, Strategy, StrategySignal


class FeatureRuleAgent(Strategy):
    """Small, explainable feature-family agent used by the committee."""

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
        atr_pct = max(0.2, float(features.get("natr_14", 1.5))) / 100.0
        risk_distance = price * min(0.08, max(0.01, atr_pct * 1.8))
        if price <= 0:
            return None
        is_buy = direction == SignalDirection.BUY
        return StrategySignal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            score=round(min(1.0, score), 3),
            confidence=round(min(1.0, score), 3),
            expected_return=round(3.0 if is_buy else 3.0, 2),
            risk_estimate=round(risk_distance / price * 100.0, 2),
            timeframe=features.get("timeframe", "1m"),
            stop_suggestion=round(price - risk_distance if is_buy else price + risk_distance, 2),
            target_suggestion=round(price + risk_distance * 2.0 if is_buy else price - risk_distance * 2.0, 2),
            market_regime=market_regime,
            reasons=reasons,
            metadata={"agent_type": self.name, "feature_family": self._description},
        )

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "timeframes": self.timeframes}


def _momentum_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    rsi = float(f.get("rsi_14", 50.0))
    macd = float(f.get("macd_hist_norm", 0.0))
    if 52 <= rsi <= 68 and macd > 0.05:
        return SignalDirection.BUY, 0.72, [f"Momentum aligned: RSI {rsi:.1f}, MACD positive"]
    if 32 <= rsi <= 48 and macd < -0.05:
        return SignalDirection.SELL, 0.72, [f"Momentum aligned: RSI {rsi:.1f}, MACD negative"]
    return SignalDirection.HOLD, 0.0, []


def _volume_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    rvol = float(f.get("rvol_20", 0.0))
    cmf = float(f.get("cmf_20", 0.0))
    if rvol >= 1.35 and cmf >= 0.10:
        return SignalDirection.BUY, min(0.9, 0.60 + cmf), [f"Volume-flow confirmation: RVOL {rvol:.2f}x, CMF {cmf:.2f}"]
    if rvol >= 1.35 and cmf <= -0.10:
        return SignalDirection.SELL, min(0.9, 0.60 + abs(cmf)), [f"Volume-flow confirmation: RVOL {rvol:.2f}x, CMF {cmf:.2f}"]
    return SignalDirection.HOLD, 0.0, []


def _regime_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    adx = float(f.get("adx_14", 0.0))
    supertrend = int(f.get("supertrend_dir", 0))
    if adx >= 22 and supertrend == 1:
        return SignalDirection.BUY, min(0.88, 0.58 + adx / 200.0), [f"Trend regime bullish: ADX {adx:.1f}, Supertrend +1"]
    if adx >= 22 and supertrend == -1:
        return SignalDirection.SELL, min(0.88, 0.58 + adx / 200.0), [f"Trend regime bearish: ADX {adx:.1f}, Supertrend -1"]
    return SignalDirection.HOLD, 0.0, []


def _vwap_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    dist = float(f.get("vwap_dist_pct", 0.0))
    rsi = float(f.get("rsi_14", 50.0))
    if dist <= -1.2 and rsi <= 38:
        return SignalDirection.BUY, 0.66, [f"VWAP reversion opportunity: distance {dist:.2f}% and RSI {rsi:.1f}"]
    if dist >= 1.2 and rsi >= 62:
        return SignalDirection.SELL, 0.66, [f"VWAP reversion opportunity: distance {dist:.2f}% and RSI {rsi:.1f}"]
    return SignalDirection.HOLD, 0.0, []


def _volatility_rule(f: dict[str, Any]) -> tuple[SignalDirection, float, list[str]]:
    squeeze = bool(f.get("is_bb_squeeze", False))
    rvol = float(f.get("rvol_20", 0.0))
    direction = int(f.get("supertrend_dir", 0))
    if squeeze and rvol >= 1.4 and direction == 1:
        return SignalDirection.BUY, 0.70, [f"Volatility breakout setup: squeeze with RVOL {rvol:.2f}x"]
    if squeeze and rvol >= 1.4 and direction == -1:
        return SignalDirection.SELL, 0.70, [f"Volatility breakdown setup: squeeze with RVOL {rvol:.2f}x"]
    return SignalDirection.HOLD, 0.0, []


def build_additional_agents() -> list[Strategy]:
    """Return five complementary agents for a ten-member committee."""
    return [
        FeatureRuleAgent("momentum_confirmation", "RSI and MACD momentum", ["1m", "5m"], _momentum_rule),
        FeatureRuleAgent("volume_flow", "RVOL and CMF institutional flow", ["1m", "5m"], _volume_rule),
        FeatureRuleAgent("regime_classifier", "ADX and Supertrend regime", ["5m", "15m"], _regime_rule),
        FeatureRuleAgent("vwap_reversion_guard", "VWAP distance and exhaustion", ["1m", "5m"], _vwap_rule),
        FeatureRuleAgent("volatility_breakout", "Bollinger squeeze and volume expansion", ["5m", "15m"], _volatility_rule),
    ]


__all__ = ["FeatureRuleAgent", "build_additional_agents"]


def build_committee_agents(existing: list[Strategy]) -> list[Strategy]:
    """Compose up to ten agents while preserving the original strategies."""
    return (existing + build_additional_agents())[:10]


__all__.append("build_committee_agents")
