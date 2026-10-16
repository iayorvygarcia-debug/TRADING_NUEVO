"""Tests for the multi-agent committee safety contract."""
from __future__ import annotations

import pytest

from strategies.base import SignalDirection
from strategies.committee_agents import build_additional_agents, build_committee_agents
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.multi_timeframe_confluence import MultiTimeframeConfluenceStrategy


@pytest.mark.anyio
async def test_committee_has_ten_distinct_agents() -> None:
    existing = [
        TrendFollowingStrategy(),
        MeanReversionStrategy(),
        BreakoutStrategy(),
        VWAPReversionStrategy(),
        MultiTimeframeConfluenceStrategy(),
    ]
    committee = build_committee_agents(existing)
    assert len(committee) == 10
    assert len({agent.name for agent in committee}) == 10


@pytest.mark.anyio
async def test_additional_agent_emits_signal_only_when_conditions_align() -> None:
    agent = build_additional_agents()[0]
    neutral = await agent.analyze("AAPL", {"close_price": 100.0, "rsi_14": 50.0, "macd_hist_norm": 0.0})
    assert neutral is None

    bullish = await agent.analyze(
        "AAPL",
        {"close_price": 100.0, "rsi_14": 60.0, "macd_hist_norm": 0.10, "natr_14": 2.0},
    )
    assert bullish is not None
    assert bullish.direction == SignalDirection.BUY
    assert bullish.stop_suggestion < bullish.target_suggestion


@pytest.mark.anyio
async def test_committee_agents_cannot_create_orders() -> None:
    agent = build_additional_agents()[1]
    signal = await agent.analyze(
        "AAPL",
        {"close_price": 100.0, "rvol_20": 2.0, "cmf_20": 0.20, "natr_14": 2.0},
    )
    assert signal is not None
    assert not hasattr(agent, "submit_order")
    assert not hasattr(agent, "broker")
