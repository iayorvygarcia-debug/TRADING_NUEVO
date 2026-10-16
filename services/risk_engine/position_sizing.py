"""AI Quant Trading Platform — Position sizing, sector exposure and market-trend filters.

Pure, side-effect-free helpers (easy to unit test) used by the worker and execution engine:

1. risk_based_shares: size = risk_$ / stop_distance, capped by max notional per position.
2. check_sector_exposure: limits concurrent positions per sector (hidden-correlation guard).
3. market_trend_multiplier: scales long/short size by benchmark trend (price vs SMA200).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from services.opportunity_engine.universe import Sector, UniverseManager


def risk_based_shares(
    capital: float,
    entry_price: float,
    stop_price: float | None,
    risk_per_trade_pct: float = 0.01,
    max_notional_pct: float = 0.20,
) -> float:
    """Shares so that a stop-out loses at most ``capital * risk_per_trade_pct``.

    Falls back to the notional cap alone when no usable stop exists. Result is floored
    to whole shares and never exceeds ``capital * max_notional_pct`` of notional.
    """
    if capital <= 0 or entry_price <= 0:
        return 0.0
    max_by_notional = math.floor((capital * max_notional_pct) / entry_price)
    if stop_price is None or stop_price <= 0:
        return float(max(0, max_by_notional))
    stop_distance = abs(entry_price - stop_price)
    if stop_distance < entry_price * 0.002:  # stop absurdly tight (<0.2%): treat as unusable
        return float(max(0, max_by_notional))
    max_by_risk = math.floor((capital * risk_per_trade_pct) / stop_distance)
    return float(max(0, min(max_by_notional, max_by_risk)))


@dataclass(frozen=True, slots=True)
class SectorCheck:
    allowed: bool
    sector: str
    open_in_sector: int
    reason: str = ""


def check_sector_exposure(
    symbol: str,
    open_symbols: Iterable[str],
    max_per_sector: int = 2,
) -> SectorCheck:
    """Reject a new entry when ``max_per_sector`` positions already sit in the same sector."""
    sector = UniverseManager.get_sector(symbol)
    if sector in (Sector.OTHER, Sector.ETF):
        return SectorCheck(True, sector.value, 0)
    count = sum(1 for s in open_symbols if s != symbol and UniverseManager.get_sector(s) == sector)
    if count >= max_per_sector:
        return SectorCheck(
            False, sector.value, count,
            f"Exposición sectorial: ya hay {count} posiciones en {sector.value} (máx {max_per_sector})",
        )
    return SectorCheck(True, sector.value, count)


def market_trend_multiplier(
    benchmark_close: float | None,
    benchmark_sma200: float | None,
    is_long: bool,
    against_trend_multiplier: float = 0.5,
) -> float:
    """Size multiplier from benchmark trend. Unknown trend -> 1.0 (never blocks on missing data).

    Benchmark above SMA200 = bull regime: longs full size, shorts reduced.
    Benchmark below SMA200 = bear regime: shorts full size, longs reduced.
    """
    if not benchmark_close or not benchmark_sma200 or benchmark_sma200 <= 0:
        return 1.0
    bull = benchmark_close >= benchmark_sma200
    if bull == is_long:
        return 1.0
    return against_trend_multiplier


def partial_exit_qty(
    quantity: float,
    entry_price: float,
    mark_price: float,
    initial_stop: float | None,
    is_long: bool,
    already_taken: bool = False,
    fraction: float = 0.5,
    trigger_r: float = 1.0,
) -> int:
    """Whole shares to sell as a partial profit-take once price reaches ``trigger_r`` x initial risk.

    Returns 0 when the partial was already taken, there is no usable initial stop, the position
    is too small to split (< 2 shares) or the trigger has not been reached. Never sells the full
    position: at least one share always remains to run with the trailing stop.
    """
    qty = abs(int(quantity))
    if already_taken or qty < 2 or entry_price <= 0 or not initial_stop:
        return 0
    risk = abs(entry_price - initial_stop)
    if risk < entry_price * 0.002:
        return 0
    gain = (mark_price - entry_price) if is_long else (entry_price - mark_price)
    if gain < risk * trigger_r:
        return 0
    return max(1, min(qty - 1, int(qty * fraction)))
