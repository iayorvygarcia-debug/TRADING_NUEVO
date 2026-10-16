"""
AI Quant Trading Platform — Shadow Trading Models & Forensic Audit Data Structures.

Provides data models for simulated execution, virtual positions, performance accounting,
MAE/MFE tracking, friction ledgers, conversion funnels, expectancy calculations,
and full forensic auditability ("¿Por qué quiso comprar AAPL?").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ShadowDecisionRecord:
    """
    Forensic record of a single trading decision evaluated by the system.
    Captures the complete audit trail from strategy to execution decision.
    Includes dual prices (signal vs execution), liquidity metrics, and slippage assumptions.
    """
    timestamp: str                               # ISO-8601 UTC string
    symbol: str                                  # Ticker symbol (e.g. "AAPL")
    strategy: str                                # Contributing strategies (e.g. "Trend Following, Breakout")
    signal: str                                  # Direction ("BUY", "SELL", "HOLD")
    action: str                                  # Action ("ENTRY", "EXIT", "REDUCE")
    side: str                                    # Order side ("BUY" or "SELL")
    session: str                                 # Market session ("REGULAR", "PREMARKET_ACTIVE", etc.)
    signal_score: float                          # Composite signal score (0.0 to 1.0)
    base_quantity: float                         # Calculated base quantity before multipliers
    session_multiplier: float                    # Multiplier based on session capabilities
    gap_multiplier: float                        # Multiplier based on GapRisk / Pre-Trade Guard
    final_quantity: float                        # Final approved quantity (0.0 if rejected)
    spread: str                                  # Spread percentage string (e.g. "0.06%")
    spread_pct: float                            # Raw spread percentage (e.g. 0.0006)
    natr: Optional[float]                        # Normalized ATR 14 (or None)
    price: float                                 # Market price at signal evaluation (signal_price)
    signal_price: float                          # Explicit signal price without friction
    shadow_execution_price: Optional[float]      # Theoretical simulated fill price
    slippage_assumption_pct: float               # Simulated slippage percentage assumption
    slippage_assumption_dollars: float           # Simulated slippage dollar impact per share
    liquidity_metric: str                        # Liquidity description (e.g. "RVOL: 1.45x" or "Normal")
    order_type: str                              # "LIMIT" or "MARKET"
    limit_price: Optional[float]                 # Limit price if applicable
    stop: Optional[float]                        # Suggested Stop-Loss price
    take_profit: Optional[float]                 # Suggested Take-Profit price
    client_order_id: str                         # Deterministic client order ID
    risk_decision: str                           # "APPROVED" or "REJECTED"
    execution_decision: str                      # "SHADOW" or "REJECTED"
    reason_for_rejection: str = "None"           # Exact violation reason(s) if rejected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_forensic_reconstruction(self) -> str:
        """
        Formats the decision into the exact forensic audit structure required by the auditor:
        "¿Por qué quiso comprar AAPL?"
        """
        lines = [
            f"Strategy:\n    {self.strategy}",
            f"Signal:\n    {self.signal}",
            f"Score:\n    {self.signal_score:.2f}",
            f"Session:\n    {self.session}",
            f"Session multiplier:\n    {self.session_multiplier:.2f}",
            f"GapRisk:\n    {'APPROVED' if self.gap_multiplier > 0 else 'REJECTED'}",
            f"Gap multiplier:\n    {self.gap_multiplier:.2f}",
            f"Spread:\n    {self.spread}",
            f"Signal Price:\n    ${self.signal_price:.2f}",
            f"Shadow Exec Price:\n    ${self.shadow_execution_price:.2f}" if self.shadow_execution_price else "Shadow Exec Price:\n    N/A (Rejected)",
            f"Slippage Assumption:\n    {self.slippage_assumption_pct * 100:.3f}% (${self.slippage_assumption_dollars:.4f}/sh)",
            f"Liquidity / RVOL:\n    {self.liquidity_metric}",
            f"Risk:\n    {self.risk_decision}",
            f"Final quantity:\n    {int(self.final_quantity) if self.final_quantity.is_integer() else self.final_quantity}",
            f"Order:\n    {self.order_type}",
            f"Client Order ID:\n    {self.client_order_id}",
            f"Execution:\n    {self.execution_decision}",
        ]
        if self.reason_for_rejection != "None":
            lines.append(f"Rejection Reason:\n    {self.reason_for_rejection}")
        return "\n\n".join(lines)


@dataclass
class ShadowOrder:
    """Theoretical order execution in shadow mode."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str                                    # "BUY" or "SELL"
    action: str                                  # "ENTRY", "EXIT", "REDUCE"
    order_type: str                              # "LIMIT" or "MARKET"
    status: str                                  # "FILLED", "REJECTED"
    quantity: float
    signal_price: float
    theoretical_entry_price: float
    simulated_slippage_pct: float
    simulated_slippage_dollars: float
    simulated_commission: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    timestamp: str
    strategies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowPosition:
    """
    Theoretical open position tracked by the shadow engine.
    Stores entry friction parameters and tracks MFE/MAE excursions dynamically.
    """
    symbol: str
    side: str                                    # "LONG" or "SHORT"
    quantity: float
    entry_price: float                           # Shadow executed entry price
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    entry_time: str
    highest_price: float
    lowest_price: float
    client_order_id: str
    strategies: list[str] = field(default_factory=list)
    signal_entry_price: float = 0.0              # Raw theoretical entry price before friction
    entry_slippage_dollars: float = 0.0          # Dollars per share lost to slippage
    entry_slippage_pct: float = 0.0
    entry_spread_pct: float = 0.0
    entry_spread_dollars: float = 0.0
    entry_commission: float = 0.0

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)

        if self.side.upper() == "LONG":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = ((price - self.entry_price) / self.entry_price) if self.entry_price > 0 else 0.0
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            self.unrealized_pnl_pct = ((self.entry_price - price) / self.entry_price) if self.entry_price > 0 else 0.0

    def update_bar(self, high: float, low: float, close: float) -> None:
        """
        Updates position using strictly subsequent post-entry bar data.
        Guarantees that MFE and MAE are updated causally with subsequent prices.
        """
        self.current_price = close
        self.highest_price = max(self.highest_price, high)
        self.lowest_price = min(self.lowest_price, low)

        if self.side.upper() == "LONG":
            self.unrealized_pnl = (close - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = ((close - self.entry_price) / self.entry_price) if self.entry_price > 0 else 0.0
        else:
            self.unrealized_pnl = (self.entry_price - close) * self.quantity
            self.unrealized_pnl_pct = ((self.entry_price - close) / self.entry_price) if self.entry_price > 0 else 0.0

    @property
    def current_mfe_pct(self) -> float:
        """Maximum Favorable Excursion percentage reached so far."""
        if self.entry_price <= 0:
            return 0.0
        if self.side.upper() == "LONG":
            return ((self.highest_price - self.entry_price) / self.entry_price) * 100.0
        return ((self.entry_price - self.lowest_price) / self.entry_price) * 100.0

    @property
    def current_mae_pct(self) -> float:
        """Maximum Adverse Excursion percentage reached so far."""
        if self.entry_price <= 0:
            return 0.0
        if self.side.upper() == "LONG":
            return ((self.lowest_price - self.entry_price) / self.entry_price) * 100.0
        return ((self.entry_price - self.highest_price) / self.entry_price) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowTrade:
    """
    Completed round-trip shadow trade for quantitative performance evaluation.
    Preserves full friction ledger (slippage, spread, commissions) and MAE/MFE.
    """
    trade_id: str
    symbol: str
    side: str                                    # "LONG" or "SHORT"
    quantity: float
    entry_price: float                           # Shadow executed entry price
    exit_price: float                            # Shadow executed exit price
    entry_time: str
    exit_time: str
    holding_seconds: float
    exit_reason: str                             # "TAKE_PROFIT", "STOP_LOSS", "SIGNAL_EXIT", "MANUAL", "EOD"
    gross_pnl: float                             # P&L before transaction friction
    total_slippage: float                        # Entry + Exit slippage in dollars
    total_commissions: float                     # Regulatory fees / commissions
    spread_friction_dollars: float               # Bid-Ask spread cost incurred
    net_pnl: float                               # Final net P&L after all frictions
    pnl_pct: float
    client_order_id: str
    strategies: list[str] = field(default_factory=list)
    signal_entry_price: float = 0.0              # Theoretical entry without friction
    signal_exit_price: float = 0.0               # Theoretical exit without friction
    entry_slippage_dollars: float = 0.0
    exit_slippage_dollars: float = 0.0
    commission_cost: float = 0.0                 # Explicit alias/breakdown for regulatory commissions
    rounding_tolerance_dollars: float = 0.0      # Explicit rounding tolerance (< $0.005)
    initial_risk_dollars: float = 0.0            # |entry_price - stop_loss_price| * quantity
    r_multiple: Optional[float] = None           # Realized R-multiple: net_pnl / initial_risk_dollars
    holding_minutes: float = 0.0                 # holding_seconds / 60.0
    mfe_pct: float = 0.0                         # Maximum Favorable Excursion % (+X.XX%)
    mae_pct: float = 0.0                         # Maximum Adverse Excursion % (-X.XX%)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyMetrics:
    """Quantitative metrics segregated per strategy."""
    signals_count: int = 0
    entries_count: int = 0
    exits_count: int = 0
    closed_trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: Optional[float] = 0.0
    profit_factor_display: str = "0.00"          # "N/A (sin pérdidas)" if loss is zero
    max_drawdown: float = 0.0
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    avg_r_multiple: float = 0.0
    expectancy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowMetrics:
    """
    Quantitative performance, risk guard audit, expectancy, MAE/MFE,
    and conversion funnel metrics with verified broker isolation.
    """
    # ── 0. Verifiable Broker Isolation Proof ──────────────────────────────────
    shadow_candidate_orders: int = 0             # Candidate orders eligible for broker execution
    shadow_gate_intercepted_orders: int = 0      # Orders intercepted by Shadow Gate (100% in shadow mode)
    real_broker_submit_calls: int = 0            # Guaranteed 0 real broker submissions
    real_broker_orders_created: int = 0          # Guaranteed 0 real broker orders

    # ── 1. Conversion Funnel (Separar señal de ejecución) ─────────────────────
    signals_generated: int = 0                   # Señales totales producidas por estrategias
    signals_approved_by_risk: int = 0            # Señales que superaron PreTradeGuard y RiskEngine
    signals_order_eligible: int = 0              # Señales aprobadas con final_quantity > 0
    virtual_entries: int = 0                     # Entradas virtuales efectivamente ejecutadas
    closed_trades_count: int = 0                 # Posiciones cerradas por TP, SL o señal

    # ── 2. Trade Counts & Win/Loss Breakdown ──────────────────────────────────
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0                        # P(win)
    loss_rate: float = 0.0                       # P(loss)
    avg_win: float = 0.0                         # Average dollar gain on winners
    avg_loss: float = 0.0                        # Average dollar loss on losers (positive number)
    expectancy: float = 0.0                      # E = P(win)*AvgWin - P(loss)*AvgLoss
    expectancy_note: str = ""                    # Disclaimer about small sample size

    # ── 3. P&L & Profit Factor (Closed Accounting Ledger) ─────────────────────
    total_gross_pnl: float = 0.0                 # Sum of gross_pnl before frictions
    total_gross_profit: float = 0.0
    total_gross_loss: float = 0.0
    profit_factor: Optional[float] = 0.0
    profit_factor_display: str = "0.00"          # "N/A (sin pérdidas)" if loss is 0
    total_net_pnl: float = 0.0
    total_slippage_paid: float = 0.0             # Accumulated slippage across all trades
    total_spread_cost_paid: float = 0.0          # Accumulated spread friction across all trades
    total_commissions_paid: float = 0.0          # Accumulated commissions paid
    unrealized_pnl_dollars: float = 0.0          # Floating unrealized P&L of open positions
    total_equity_change: float = 0.0             # Realized Net P&L + Unrealized Floating P&L

    # ── 4. Equity Curve & Drawdown ────────────────────────────────────────────
    max_drawdown_dollars: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_virtual_equity: float = 10000.0
    current_virtual_equity: float = 10000.0

    # ── 5. Excursions & Risk Multiples (MFE / MAE / R-Multiple / Duration) ───
    avg_mfe_pct: float = 0.0                     # System-wide average Maximum Favorable Excursion
    avg_mae_pct: float = 0.0                     # System-wide average Maximum Adverse Excursion
    avg_r_multiple: float = 0.0                  # System-wide average R-multiple realized
    median_r_multiple: float = 0.0               # System-wide median R-multiple realized
    cumulative_r_multiple: float = 0.0           # Sum of all realized R-multiples
    expectancy_r: float = 0.0                    # Mathematical expectancy in R units
    std_dev_r: float = 0.0                       # Standard deviation of R-multiples
    max_r: float = 0.0                           # Maximum realized R-multiple
    min_r: float = 0.0                           # Minimum realized R-multiple
    avg_holding_minutes: float = 0.0             # System-wide average trade duration in minutes
    median_holding_minutes: float = 0.0          # System-wide median trade duration in minutes
    duration_source: str = "BAR_SIMULATION"      # "BAR_SIMULATION" (holding_minutes=0.0 when no intrabar timestamp)
    r_distribution: dict[str, int] = field(default_factory=dict)

    # ── 6. Slippage & Spread Empirical Distributions ──────────────────────────
    slippage_avg_pct: float = 0.0
    slippage_median_pct: float = 0.0
    slippage_max_pct: float = 0.0
    spread_avg_pct: float = 0.0
    spread_median_pct: float = 0.0

    # ── 7. Segregated Breakdowns ──────────────────────────────────────────────
    signals_by_strategy: dict[str, int] = field(default_factory=dict)
    rejections_by_guard: dict[str, int] = field(default_factory=dict)
    rejection_reasons: list[dict[str, Any]] = field(default_factory=list)
    strategy_metrics: dict[str, StrategyMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        if "strategy_metrics" in res:
            res["strategy_metrics"] = {k: v for k, v in res["strategy_metrics"].items()}
        return res
