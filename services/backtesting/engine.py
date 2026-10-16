"""
AI Quant Trading Platform — Event-Driven Backtesting Engine.

Simulates real-world algorithmic execution bar-by-bar:
- Strict chronological causality (no lookahead bias)
- Intrabar Stop-Loss and Take-Profit execution
- Realistic transaction costs (spread, slippage, commissions, SEC fees)
- Risk-based position sizing (Fixed % risk of equity to stop-loss)
- Generates equity curves, trade logs, and comprehensive quantitative metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from core.logging import get_logger
from services.backtesting.cost_model import CostModel
from services.backtesting.metrics import MetricsCalculator, PerformanceMetrics
from services.backtesting.portfolio import SimulatedPortfolio, SimulatedTrade
from services.strategy_engine.signal_combiner import CombinedSignal, SignalCombiner
from services.technical_analysis.feature_engine import FeatureEngine
from strategies.base import SignalDirection, Strategy

logger = get_logger("backtest_engine")


@dataclass
class BacktestResult:
    """Consolidated backtest output."""
    metrics: PerformanceMetrics
    trades: list[SimulatedTrade]
    equity_curve: pd.DataFrame
    config: dict[str, Any]

    def summary(self) -> str:
        """Formatted textual report for console or artifacts."""
        m = self.metrics
        return (
            "==========================================================\n"
            "               BACKTEST PERFORMANCE REPORT                \n"
            "==========================================================\n"
            f"Initial Capital:       ${m.initial_capital:,.2f}\n"
            f"Final Equity:          ${m.final_equity:,.2f}\n"
            f"Total Return:          {m.total_return_pct:+.2f}%\n"
            f"CAGR:                  {m.cagr_pct:+.2f}%\n"
            "----------------------------------------------------------\n"
            f"Sharpe Ratio:          {m.sharpe_ratio:.2f}\n"
            f"Sortino Ratio:         {m.sortino_ratio:.2f}\n"
            f"Max Drawdown:          {m.max_drawdown_pct:.2f}%\n"
            f"Calmar Ratio:          {m.calmar_ratio:.2f}\n"
            "----------------------------------------------------------\n"
            f"Total Trades:          {m.total_trades}\n"
            f"Win Rate:              {m.win_rate_pct:.1f}%\n"
            f"Profit Factor:         {m.profit_factor:.2f}\n"
            f"Average Trade Return:  {m.avg_trade_pct:+.2f}%\n"
            f"Expectancy:            ${m.expectancy:+.2f} per trade\n"
            "=========================================================="
        )


class EventDrivenBacktester:
    """
    High-fidelity event-driven backtesting engine.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_per_trade_pct: float = 2.0,       # 2% equity risked per trade
        max_position_size_pct: float = 20.0,   # Max 20% of equity in a single symbol
        cost_model: CostModel | None = None,
        signal_combiner: SignalCombiner | None = None,
        warmup_bars: int = 50,
    ) -> None:
        self.initial_capital = initial_capital
        self.risk_pct = risk_per_trade_pct / 100.0
        self.max_pos_pct = max_position_size_pct / 100.0
        self.cost_model = cost_model or CostModel()
        self.combiner = signal_combiner or SignalCombiner()
        self.warmup_bars = warmup_bars

    async def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategies: list[Strategy],
        timeframe: str = "1d",
    ) -> BacktestResult:
        """
        Execute full backtest on historical OHLCV data.
        """
        if len(df) <= self.warmup_bars:
            raise ValueError(
                f"Data length ({len(df)}) must exceed warmup_bars ({self.warmup_bars})"
            )

        portfolio = SimulatedPortfolio(initial_capital=self.initial_capital)
        n = len(df)

        logger.info(
            "backtest_started",
            symbol=symbol,
            bars=n,
            strategies=[s.name for s in strategies],
            initial_capital=self.initial_capital,
        )

        for i in range(self.warmup_bars, n):
            current_bar = df.iloc[i]
            # Window of historical data up to current bar (NO LOOKAHEAD)
            history_window = df.iloc[: i + 1]

            # Parse bar timestamp
            if isinstance(current_bar.name, (pd.Timestamp, datetime)):
                ts = pd.to_datetime(current_bar.name).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            elif "time" in current_bar:
                ts = pd.to_datetime(current_bar["time"]).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            open_p = float(current_bar["open"])
            high_p = float(current_bar["high"])
            low_p = float(current_bar["low"])
            close_p = float(current_bar["close"])

            # ── 1. Intrabar Stop-Loss / Take-Profit Checking ──────────
            if symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                exited = False

                if pos.is_long:
                    # Check stop loss hit
                    if pos.stop_loss and low_p <= pos.stop_loss:
                        fill = self.cost_model.calculate_fill_price(pos.stop_loss, is_buy=False)
                        comm = self.cost_model.calculate_commission(pos.quantity, fill, is_buy=False)
                        portfolio.close_position(symbol, fill, ts, comm, exit_reason="stop_loss")
                        exited = True
                    # Check take profit hit
                    elif pos.take_profit and high_p >= pos.take_profit:
                        fill = self.cost_model.calculate_fill_price(pos.take_profit, is_buy=False)
                        comm = self.cost_model.calculate_commission(pos.quantity, fill, is_buy=False)
                        portfolio.close_position(symbol, fill, ts, comm, exit_reason="take_profit")
                        exited = True
                else:
                    # Short position
                    if pos.stop_loss and high_p >= pos.stop_loss:
                        fill = self.cost_model.calculate_fill_price(pos.stop_loss, is_buy=True)
                        comm = self.cost_model.calculate_commission(pos.quantity, fill, is_buy=True)
                        portfolio.close_position(symbol, fill, ts, comm, exit_reason="stop_loss")
                        exited = True
                    elif pos.take_profit and low_p <= pos.take_profit:
                        fill = self.cost_model.calculate_fill_price(pos.take_profit, is_buy=True)
                        comm = self.cost_model.calculate_commission(pos.quantity, fill, is_buy=True)
                        portfolio.close_position(symbol, fill, ts, comm, exit_reason="take_profit")
                        exited = True

                if not exited:
                    portfolio.update_prices({symbol: close_p})

            # ── 2. Feature Extraction ────────────────────────────────
            fv = FeatureEngine.compute_features(
                history_window,
                symbol=symbol,
                timeframe=timeframe,
            )
            features_dict = fv.to_dict()

            # ── 3. Strategy Signals Generation ───────────────────────
            signals = []
            for strat in strategies:
                sig = await strat.analyze(symbol, features_dict)
                if sig:
                    signals.append(sig)

            # ── 4. Combine Signals ───────────────────────────────────
            combined = self.combiner.combine(symbol, signals)

            # ── 5. Position Management & Execution ───────────────────
            if symbol not in portfolio.positions:
                if combined.is_actionable:
                    self._execute_entry(portfolio, combined, close_p, ts)
            else:
                # Opposite signal can trigger early exit
                pos = portfolio.positions[symbol]
                if pos.is_long and combined.direction == SignalDirection.SELL:
                    fill = self.cost_model.calculate_fill_price(close_p, is_buy=False)
                    comm = self.cost_model.calculate_commission(pos.quantity, fill, is_buy=False)
                    portfolio.close_position(symbol, fill, ts, comm, exit_reason="signal_reversal")
                elif not pos.is_long and combined.direction == SignalDirection.BUY:
                    fill = self.cost_model.calculate_fill_price(close_p, is_buy=True)
                    comm = self.cost_model.calculate_commission(pos.quantity, fill, is_buy=True)
                    portfolio.close_position(symbol, fill, ts, comm, exit_reason="signal_reversal")

            # ── 6. Record Daily/Bar Snapshot ─────────────────────────
            portfolio.update_prices({symbol: close_p})
            portfolio.record_snapshot(ts)

        # ── End of Backtest: Close remaining open positions ──────────
        last_bar = df.iloc[-1]
        last_close = float(last_bar["close"])
        if symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            is_buy = not pos.is_long
            fill = self.cost_model.calculate_fill_price(last_close, is_buy=is_buy)
            comm = self.cost_model.calculate_commission(pos.quantity, fill, is_buy=is_buy)
            portfolio.close_position(symbol, fill, ts, comm, exit_reason="end_of_backtest")

        # ── 7. Calculate Comprehensive Performance Metrics ───────────
        eq_df = pd.DataFrame(portfolio.equity_history)
        if not eq_df.empty:
            eq_df.set_index("timestamp", inplace=True)
            equity_series = eq_df["equity"]
        else:
            equity_series = pd.Series([self.initial_capital])

        trade_dicts = [t.to_dict() for t in portfolio.closed_trades]
        metrics = MetricsCalculator.calculate(
            equity_curve=equity_series,
            trades=trade_dicts,
            initial_capital=self.initial_capital,
        )

        logger.info(
            "backtest_completed",
            symbol=symbol,
            total_trades=metrics.total_trades,
            win_rate=f"{metrics.win_rate_pct:.1f}%",
            profit_factor=f"{metrics.profit_factor:.2f}",
            total_return=f"{metrics.total_return_pct:.2f}%",
            max_drawdown=f"{metrics.max_drawdown_pct:.2f}%",
            sharpe=f"{metrics.sharpe_ratio:.2f}",
        )

        return BacktestResult(
            metrics=metrics,
            trades=portfolio.closed_trades,
            equity_curve=eq_df,
            config={
                "symbol": symbol,
                "initial_capital": self.initial_capital,
                "risk_per_trade_pct": self.risk_pct * 100.0,
                "max_position_size_pct": self.max_pos_pct * 100.0,
                "strategies": [s.name for s in strategies],
            },
        )

    def _execute_entry(
        self,
        portfolio: SimulatedPortfolio,
        signal: CombinedSignal,
        current_price: float,
        timestamp: datetime,
    ) -> None:
        """Position sizing and order execution."""
        current_equity = portfolio.equity
        max_notional = current_equity * self.max_pos_pct
        risk_dollar_amount = current_equity * self.risk_pct

        is_buy = signal.direction == SignalDirection.BUY
        fill_price = self.cost_model.calculate_fill_price(current_price, is_buy=is_buy)

        # Risk-based sizing: Shares = Risk Amount / (Price - Stop)
        if signal.suggested_stop:
            risk_per_share = abs(fill_price - signal.suggested_stop)
            if risk_per_share > 1e-4:
                raw_shares = risk_dollar_amount / risk_per_share
            else:
                raw_shares = max_notional / fill_price
        else:
            raw_shares = max_notional / fill_price

        # Cap by maximum position allocation
        max_shares_by_cap = max_notional / fill_price
        shares = min(raw_shares, max_shares_by_cap)
        shares = max(1.0, float(int(shares)))  # Integer shares

        comm = self.cost_model.calculate_commission(shares, fill_price, is_buy=is_buy)
        strat_name = signal.contributing_strategies[0] if signal.contributing_strategies else "ensemble"

        portfolio.open_position(
            symbol=signal.symbol,
            direction=signal.direction,
            quantity=shares,
            price=fill_price,
            timestamp=timestamp,
            stop_loss=signal.suggested_stop,
            take_profit=signal.suggested_target,
            commission=comm,
            strategy=strat_name,
        )
