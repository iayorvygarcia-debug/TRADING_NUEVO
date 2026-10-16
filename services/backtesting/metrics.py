"""
AI Quant Trading Platform — Quantitative Performance Metrics Calculator.

Computes standard financial & quant metrics from backtest equity curves and trade logs:
- Sharpe, Sortino, Calmar ratios
- Max Drawdown and Drawdown durations
- Win rate, Profit Factor, Expectancy
- Annualized return (CAGR)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PerformanceMetrics:
    """Comprehensive performance metrics summary."""
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    profit_factor: float
    win_rate_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float
    avg_trade_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "cagr_pct": round(self.cagr_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "profit_factor": round(self.profit_factor, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_trade_pnl": round(self.avg_trade_pnl, 2),
            "avg_trade_pct": round(self.avg_trade_pct, 2),
            "avg_win_pct": round(self.avg_win_pct, 2),
            "avg_loss_pct": round(self.avg_loss_pct, 2),
            "expectancy": round(self.expectancy, 2),
        }


class MetricsCalculator:
    """Calculates all portfolio and trade performance statistics."""

    @classmethod
    def calculate(
        cls,
        equity_curve: pd.Series,
        trades: list[dict[str, Any]],
        initial_capital: float,
        risk_free_rate: float = 0.04,  # 4% annual risk-free rate
        annual_periods: float = 252.0,  # 252 trading days/year
    ) -> PerformanceMetrics:
        """
        Compute all performance metrics.
        
        Args:
            equity_curve: Series of portfolio equity over time.
            trades: List of completed trade dictionaries.
            initial_capital: Starting cash balance.
            risk_free_rate: Annualized risk-free benchmark.
            annual_periods: Number of periods per year for annualization.
        """
        final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital
        total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

        # Returns series
        rets = equity_curve.pct_change().dropna()
        n_periods = len(rets)

        # CAGR
        years = max(1.0 / annual_periods, n_periods / annual_periods)
        cagr_pct = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

        # Sharpe Ratio
        if len(rets) > 1 and rets.std() > 1e-8:
            rf_per_period = (1.0 + risk_free_rate) ** (1.0 / annual_periods) - 1.0
            excess_returns = rets - rf_per_period
            sharpe = (excess_returns.mean() / rets.std()) * math.sqrt(annual_periods)
        else:
            sharpe = 0.0

        # Sortino Ratio (Downside deviation only)
        downside_rets = rets[rets < 0]
        if len(downside_rets) > 1 and downside_rets.std() > 1e-8:
            rf_per_period = (1.0 + risk_free_rate) ** (1.0 / annual_periods) - 1.0
            excess_returns = rets - rf_per_period
            sortino = (excess_returns.mean() / downside_rets.std()) * math.sqrt(annual_periods)
        else:
            sortino = 0.0

        # Max Drawdown
        cum_max = equity_curve.cummax()
        drawdown = (equity_curve - cum_max) / (cum_max + 1e-12)
        max_dd_pct = abs(float(drawdown.min())) * 100.0 if not drawdown.empty else 0.0

        # Calmar Ratio
        calmar = (cagr_pct / max_dd_pct) if max_dd_pct > 1e-4 else 0.0

        # Trade-level statistics
        total_trades = len(trades)
        if total_trades > 0:
            pnls = [t["net_pnl"] for t in trades]
            pnl_pcts = [t["return_pct"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            win_pcts = [pct for pct in pnl_pcts if pct > 0]
            loss_pcts = [pct for pct in pnl_pcts if pct <= 0]

            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate_pct = (winning_trades / total_trades) * 100.0

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

            avg_trade_pnl = sum(pnls) / total_trades
            avg_trade_pct = sum(pnl_pcts) / total_trades
            avg_win_pct = sum(win_pcts) / len(win_pcts) if win_pcts else 0.0
            avg_loss_pct = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0

            # Expectancy: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
            win_rate = winning_trades / total_trades
            loss_rate = losing_trades / total_trades
            avg_win_dollar = (sum(wins) / len(wins)) if wins else 0.0
            avg_loss_dollar = (abs(sum(losses)) / len(losses)) if losses else 0.0
            expectancy = (win_rate * avg_win_dollar) - (loss_rate * avg_loss_dollar)
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate_pct = 0.0
            profit_factor = 0.0
            avg_trade_pnl = 0.0
            avg_trade_pct = 0.0
            avg_win_pct = 0.0
            avg_loss_pct = 0.0
            expectancy = 0.0

        return PerformanceMetrics(
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            cagr_pct=cagr_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=calmar,
            profit_factor=profit_factor,
            win_rate_pct=win_rate_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_trade_pnl=avg_trade_pnl,
            avg_trade_pct=avg_trade_pct,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            expectancy=expectancy,
        )
