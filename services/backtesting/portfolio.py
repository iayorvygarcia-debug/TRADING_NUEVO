"""
AI Quant Trading Platform — Simulated Backtesting Portfolio.

Maintains ledger of cash, open positions, trade history, and equity tracking.
Supports:
- Long and Short positions
- Stop-Loss and Take-Profit automated order fills
- Commission deduction and realized PnL accounting
- Equity mark-to-market curve generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from strategies.base import SignalDirection


@dataclass
class SimulatedPosition:
    symbol: str
    direction: SignalDirection      # BUY (Long) or SELL (Short)
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    current_price: float = 0.0
    strategy: str = ""

    @property
    def is_long(self) -> bool:
        return self.direction == SignalDirection.BUY

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.is_long:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.is_long:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100.0
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100.0


@dataclass
class SimulatedTrade:
    symbol: str
    direction: str                  # "long" or "short"
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    net_pnl: float
    return_pct: float
    commission: float
    exit_reason: str                # "stop_loss", "take_profit", "signal_exit", "end_of_backtest"
    strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "entry_time": self.entry_time.isoformat() if hasattr(self.entry_time, "isoformat") else str(self.entry_time),
            "exit_time": self.exit_time.isoformat() if hasattr(self.exit_time, "isoformat") else str(self.exit_time),
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round(self.return_pct, 2),
            "commission": round(self.commission, 2),
            "exit_reason": self.exit_reason,
            "strategy": self.strategy,
        }


class SimulatedPortfolio:
    """Simulated accounting engine for event-driven backtesting."""

    def __init__(self, initial_capital: float = 100000.0) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, SimulatedPosition] = {}
        self.closed_trades: list[SimulatedTrade] = []
        self.equity_history: list[dict[str, Any]] = []

    @property
    def total_market_value(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def equity(self) -> float:
        """Total equity = Cash + Market Value (or cash + unrealized pnl)."""
        # For long: cash was deducted at entry, so equity = cash + market_value
        # For short: cash was added at entry (borrow proceeds), so equity = cash - market_value
        eq = self.cash
        for pos in self.positions.values():
            if pos.is_long:
                eq += pos.market_value
            else:
                # Liability to buy back shares
                eq -= pos.market_value
        return eq

    def open_position(
        self,
        symbol: str,
        direction: SignalDirection,
        quantity: float,
        price: float,
        timestamp: datetime,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        commission: float = 0.0,
        strategy: str = "",
    ) -> bool:
        """Open a new long or short position."""
        cost = quantity * price
        total_required = cost + commission

        if direction == SignalDirection.BUY:
            if self.cash < total_required:
                return False  # Insufficient cash
            self.cash -= total_required
        else:
            # Short position: cash receives sale proceeds minus commission, but margin required
            if self.cash < cost * 0.5:  # Require 50% margin
                return False
            self.cash += cost - commission

        pos = SimulatedPosition(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=price,
            strategy=strategy,
        )
        self.positions[symbol] = pos
        return True

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
        exit_reason: str = "signal_exit",
    ) -> SimulatedTrade | None:
        """Close an existing open position and realize PnL."""
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None

        if pos.is_long:
            gross_pnl = (price - pos.entry_price) * pos.quantity
            net_pnl = gross_pnl - commission
            return_pct = ((price - pos.entry_price) / pos.entry_price) * 100.0
            # Cash returns from selling shares
            self.cash += (pos.quantity * price) - commission
        else:
            gross_pnl = (pos.entry_price - price) * pos.quantity
            net_pnl = gross_pnl - commission
            return_pct = ((pos.entry_price - price) / pos.entry_price) * 100.0
            # Buy back shares to cover short
            self.cash -= (pos.quantity * price) + commission

        trade = SimulatedTrade(
            symbol=symbol,
            direction="long" if pos.is_long else "short",
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=price,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=return_pct,
            commission=commission,
            exit_reason=exit_reason,
            strategy=pos.strategy,
        )
        self.closed_trades.append(trade)
        return trade

    def update_prices(self, price_map: dict[str, float]) -> None:
        """Update current market price of all positions."""
        for sym, price in price_map.items():
            if sym in self.positions:
                self.positions[sym].current_price = price

    def record_snapshot(self, timestamp: datetime) -> None:
        """Record point-in-time equity snapshot."""
        self.equity_history.append({
            "timestamp": timestamp,
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "open_positions": len(self.positions),
            "unrealized_pnl": round(self.total_unrealized_pnl, 2),
        })
