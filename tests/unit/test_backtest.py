"""
Unit and Integration Tests for Event-Driven Backtesting.

Validates:
1. CostModel (spread, slippage, commission, SEC/FINRA fees)
2. SimulatedPortfolio (ledger, positions, mark-to-market equity)
3. MetricsCalculator (Sharpe, Sortino, Drawdown, Profit Factor, Expectancy)
4. EventDrivenBacktester (end-to-end bar simulation with strategies)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest

from services.backtesting.cost_model import CostModel
from services.backtesting.metrics import MetricsCalculator, PerformanceMetrics
from services.backtesting.portfolio import SimulatedPortfolio
from services.backtesting.engine import EventDrivenBacktester, BacktestResult
from strategies.base import SignalDirection
from strategies.trend_following import TrendFollowingStrategy


class TestCostModel:

    def test_fill_prices(self) -> None:
        model = CostModel(half_spread_bps=2.0, slippage_bps=3.0)  # Total 5 bps friction = 0.05%
        base = 100.0

        buy_fill = model.calculate_fill_price(base, is_buy=True)
        sell_fill = model.calculate_fill_price(base, is_buy=False)

        assert buy_fill > base
        assert sell_fill < base
        assert pytest.approx(buy_fill, abs=1e-3) == 100.05
        assert pytest.approx(sell_fill, abs=1e-3) == 99.95

    def test_commissions(self) -> None:
        model = CostModel()
        # Buy: $0 commission on Alpaca
        assert model.calculate_commission(100, 150.0, is_buy=True) == 0.0

        # Sell: SEC fee + FINRA TAF applies
        sell_comm = model.calculate_commission(100, 150.0, is_buy=False)
        assert sell_comm > 0.0


class TestSimulatedPortfolio:

    def test_open_and_close_long(self) -> None:
        pf = SimulatedPortfolio(initial_capital=50000.0)
        t1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)

        # Open Long: 100 shares @ $150 = $15,000
        success = pf.open_position("AAPL", SignalDirection.BUY, quantity=100, price=150.0, timestamp=t1)
        assert success is True
        assert pf.cash == 35000.0
        assert pf.equity == 50000.0
        assert "AAPL" in pf.positions

        # Close Long @ $160 (+$10/share = +$1,000 profit)
        trade = pf.close_position("AAPL", price=160.0, timestamp=t2)
        assert trade is not None
        assert trade.net_pnl == 1000.0
        assert trade.return_pct == pytest.approx(6.666, abs=1e-2)
        assert pf.cash == 51000.0
        assert pf.equity == 51000.0
        assert "AAPL" not in pf.positions


class TestMetricsCalculator:

    def test_metrics_calculation(self) -> None:
        # 100 days of equity increasing from 100,000 to 115,000
        days = pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC")
        curve = pd.Series(np.linspace(100000, 115000, 100), index=days)

        trades = [
            {"net_pnl": 500.0, "return_pct": 5.0},
            {"net_pnl": 1000.0, "return_pct": 10.0},
            {"net_pnl": -300.0, "return_pct": -3.0},
            {"net_pnl": 800.0, "return_pct": 8.0},
        ]

        metrics = MetricsCalculator.calculate(
            equity_curve=curve,
            trades=trades,
            initial_capital=100000.0,
        )

        assert metrics.total_return_pct == 15.0
        assert metrics.total_trades == 4
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 1
        assert metrics.win_rate_pct == 75.0
        assert metrics.profit_factor > 2.0
        assert metrics.expectancy > 0.0


class TestEventDrivenBacktester:

    @pytest.fixture
    def trending_ohlcv(self) -> pd.DataFrame:
        """120 bars of an upward trending market."""
        np.random.seed(123)
        n = 120
        start = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
        times = [start + timedelta(days=i) for i in range(n)]

        # Upward drift
        drift = np.linspace(100, 160, n)
        noise = np.random.normal(0, 1.0, n)
        closes = drift + noise
        highs = closes + np.random.uniform(0.5, 2.0, n)
        lows = closes - np.random.uniform(0.5, 2.0, n)
        opens = (highs + lows) / 2.0
        volumes = np.random.uniform(50000, 150000, n)

        df = pd.DataFrame({
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })
        df.set_index("time", inplace=True)
        return df

    @pytest.mark.anyio
    async def test_run_backtest(self, trending_ohlcv: pd.DataFrame) -> None:
        strategy = TrendFollowingStrategy(min_adx=15.0)
        backtester = EventDrivenBacktester(
            initial_capital=100000.0,
            risk_per_trade_pct=2.0,
            warmup_bars=40,
        )

        result = await backtester.run(
            df=trending_ohlcv,
            symbol="SPY",
            strategies=[strategy],
            timeframe="1d",
        )

        assert isinstance(result, BacktestResult)
        assert result.metrics.initial_capital == 100000.0
        assert not result.equity_curve.empty
        assert "equity" in result.equity_curve.columns
        assert isinstance(result.summary(), str)
        assert "BACKTEST PERFORMANCE REPORT" in result.summary()
