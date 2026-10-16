"""Calibración por backtest de los umbrales de entrada (score compuesto y consenso).

Descarga ~2 años de velas diarias de Alpaca, calcula features y señales UNA vez por barra
(no dependen de los umbrales) y luego barre una rejilla de (min_composite_score, min_consensus_ratio)
con el EventDrivenBacktester.

Uso:  uv run python scripts/calibrate_thresholds.py [--symbols AAPL,MSFT,...] [--days 700]
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

import services.backtesting.engine as bt_engine  # noqa: E402
from core.config import get_settings  # noqa: E402
from services.backtesting.engine import EventDrivenBacktester  # noqa: E402
from services.strategy_engine.signal_combiner import SignalCombiner  # noqa: E402
from services.technical_analysis.feature_engine import FeatureEngine  # noqa: E402
from strategies.base import Strategy  # noqa: E402
from strategies.breakout import BreakoutStrategy  # noqa: E402
from strategies.mean_reversion import MeanReversionStrategy  # noqa: E402
from strategies.multi_timeframe_confluence import MultiTimeframeConfluenceStrategy  # noqa: E402
from strategies.trend_following import TrendFollowingStrategy  # noqa: E402
from strategies.vwap_reversion import VWAPReversionStrategy  # noqa: E402

DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AVGO,AMD,NFLX,ADBE,CRWD,PYPL,QCOM,INTU,COST"


class _CachedFeatureEngine:
    """Memoises FeatureEngine.compute_features so each bar is computed once across the whole grid."""

    _cache: dict[tuple, object] = {}

    @classmethod
    def compute_features(cls, df, symbol, timeframe="1d", mtf_alignment=None):
        key = (symbol, len(df), str(df.index[-1]))
        if key not in cls._cache:
            cls._cache[key] = FeatureEngine.compute_features(df, symbol=symbol, timeframe=timeframe)
        return cls._cache[key]


class _CachedStrategy(Strategy):
    """Memoises a strategy's signal per (symbol, close, timestamp)."""

    def __init__(self, inner: Strategy):
        self._inner = inner
        self._memo: dict[tuple, object] = {}

    name = property(lambda self: self._inner.name)
    description = property(lambda self: self._inner.description)
    timeframes = property(lambda self: self._inner.timeframes)

    def get_config(self):
        return self._inner.get_config()

    async def analyze(self, symbol, features, market_regime=None):
        key = (symbol, features.get("timestamp"), features.get("close_price"))
        if key not in self._memo:
            self._memo[key] = await self._inner.analyze(symbol, features, market_regime)
        return self._memo[key]


def fetch_bars(symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    s = get_settings()
    sec = s.alpaca.secret_key
    raw_sec = sec.get_secret_value() if hasattr(sec, "get_secret_value") else str(sec)
    client = StockHistoricalDataClient(s.alpaca.api_key, raw_sec)
    req = StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=days), adjustment="all",
    )
    raw = client.get_stock_bars(req).df
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym in raw.index.get_level_values(0):
            d = raw.loc[sym][["open", "high", "low", "close", "volume"]].copy()
            d.index = pd.to_datetime(d.index)
            out[sym] = d
    return out


async def run_grid(data: dict[str, pd.DataFrame], scores: list[float], consensus: list[float], min_voters: int):
    strategies = [_CachedStrategy(s) for s in (
        TrendFollowingStrategy(), MeanReversionStrategy(), BreakoutStrategy(),
        VWAPReversionStrategy(), MultiTimeframeConfluenceStrategy(),
    )]
    bt_engine.FeatureEngine = _CachedFeatureEngine  # type: ignore[assignment]
    rows = []
    for sc, cr in itertools.product(scores, consensus):
        trades = wins = 0
        gross_win = gross_loss = 0.0
        rets: list[float] = []
        for sym, df in data.items():
            combiner = SignalCombiner(min_composite_score=sc, min_consensus_ratio=cr, min_voters=min_voters)
            bt = EventDrivenBacktester(initial_capital=100000.0, risk_per_trade_pct=1.0, signal_combiner=combiner)
            try:
                res = await bt.run(df, sym, strategies)
            except Exception as exc:  # insufficient data etc.
                print(f"  [skip] {sym}: {exc}")
                continue
            for t in res.trades:
                d = t.to_dict()
                pnl = float(d.get("net_pnl", d.get("pnl", 0.0)))
                trades += 1
                wins += pnl > 0
                gross_win += max(0.0, pnl)
                gross_loss += max(0.0, -pnl)
                rets.append(pnl)
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
        rows.append({
            "min_score": sc, "min_consensus": cr, "trades": trades,
            "win_rate_%": round(100 * wins / trades, 1) if trades else 0.0,
            "profit_factor": round(pf, 2),
            "expectancy_$": round(sum(rets) / trades, 2) if trades else 0.0,
            "net_pnl_$": round(sum(rets), 2),
        })
        print(f"  score={sc:.2f} consensus={cr:.2f} -> trades={trades} pf={rows[-1]['profit_factor']} pnl={rows[-1]['net_pnl_$']}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--days", type=int, default=700)
    ap.add_argument("--min-voters", type=int, default=2)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"Descargando {len(symbols)} símbolos ({args.days} días)...")
    data = fetch_bars(symbols, args.days)
    print(f"Datos listos: {len(data)} símbolos. Barriendo rejilla (la primera pasada calcula features)...")

    table = asyncio.run(run_grid(
        data,
        scores=[0.55, 0.60, 0.65, 0.70, 0.75],
        consensus=[0.40, 0.50, 0.60, 0.70],
        min_voters=args.min_voters,
    ))
    print("\n=== RESULTADOS (ordenados por profit factor, mínimo 20 trades) ===")
    ranked = table[table["trades"] >= 20].sort_values(["profit_factor", "net_pnl_$"], ascending=False)
    print(ranked.to_string(index=False) if not ranked.empty else table.to_string(index=False))
    out = Path("data") / "calibration_results.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
