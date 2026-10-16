"""
AI Quant Trading Platform — Broker Interface

Abstract base class for all broker implementations.
This is the critical abstraction that separates strategy from execution.

Architecture:
    BrokerInterface (this file)
    ├── PaperBroker      (FASE 8 — simulated execution)
    └── AlpacaBroker     (FASE 16 — live execution via Alpaca API)

CRITICAL RULE:
    Strategy → Signal → Portfolio → Risk → Execution → BrokerInterface → Broker
    
    The strategy NEVER has direct access to the broker.
    If Risk Engine rejects an operation, NO other layer can bypass it.

The same business logic runs for PAPER and LIVE.
Only the BrokerInterface implementation changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Order Enums ─────────────────────────────

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderAction(str, Enum):
    """Explicit position intention, differentiating entries from exits/reductions."""
    ENTRY = "entry"       # Opening or increasing a position
    EXIT = "exit"         # Closing an entire position
    REDUCE = "reduce"     # Partial close / trimming position
    REVERSE = "reverse"   # Closing opposite position and entering reverse


class BrokerHealthStatus(str, Enum):
    """Broker connectivity and state availability."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    STATE_UNKNOWN = "state_unknown"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderClass(str, Enum):
    SIMPLE = "simple"
    BRACKET = "bracket"
    OCO = "oco"
    OTO = "oto"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"        # Good til cancelled
    IOC = "ioc"        # Immediate or cancel
    FOK = "fok"        # Fill or kill
    OPG = "opg"        # Market on open
    CLS = "cls"        # Market on close


class OrderStatus(str, Enum):
    """Order state machine states."""
    CREATED = "created"
    VALIDATING = "validating"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


# ── Order Models ────────────────────────────

class OrderRequest(BaseModel):
    """Request to place an order. Created by Execution Engine."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    action: OrderAction = OrderAction.ENTRY  # Explicit intention (ENTRY, EXIT, REDUCE, REVERSE)
    client_order_id: str | None = None      # Deterministic idempotency key for multi-worker safety
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    order_class: OrderClass = OrderClass.SIMPLE
    extended_hours: bool = False

    # Bracket order legs
    take_profit_price: float | None = None
    stop_loss_price: float | None = None

    # Metadata (not sent to broker)
    signal_id: str | None = None
    strategy: str | None = None


class OrderResult(BaseModel):
    """Result from broker after order submission."""
    broker_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    quantity: float
    filled_qty: float = 0
    filled_avg_price: float | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    error_message: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class Position(BaseModel):
    """Current position from broker."""
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class AccountInfo(BaseModel):
    """Broker account information."""
    account_id: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    trading_mode: str  # "paper" or "live"


# ── Broker Interface ────────────────────────

class BrokerInterface(ABC):
    """
    Abstract interface for broker implementations.
    
    PAPER and LIVE brokers both implement this.
    The platform does not know (or care) which one is active.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker name (e.g., 'paper', 'alpaca_live')."""
        ...

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Whether this is a paper/simulated broker."""
        ...

    # ── Lifecycle ───────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the broker. Validate credentials."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        ...

    # ── Account ─────────────────────────────

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get current account information."""
        ...

    # ── Orders ──────────────────────────────

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """
        Submit an order to the broker.
        
        This is the ONLY way orders reach the market.
        Must be called ONLY by the Execution Engine,
        which is called ONLY after Risk Engine approval.
        """
        ...

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> OrderResult:
        """Cancel a pending order."""
        ...

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> OrderResult:
        """Get current status of an order."""
        ...

    @abstractmethod
    async def get_open_orders(self) -> list[OrderResult]:
        """Get all open/pending orders."""
        ...

    async def get_order_by_client_order_id(self, client_order_id: str) -> OrderResult | None:
        """Get order by client_order_id if supported by the broker implementation."""
        return None

    # ── Positions ───────────────────────────

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all current positions."""
        ...

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""
        ...

    # ── Reconciliation ──────────────────────

    @abstractmethod
    async def reconcile(self) -> dict[str, Any]:
        """
        Reconcile local state with broker state.
        
        Compare positions, orders, and account info.
        Report discrepancies.
        
        Returns:
            Dict with reconciliation results and any discrepancies found.
        """
        ...

    # ── Health ──────────────────────────────

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check broker connection health."""
        ...

    # ── History & Reset ─────────────────────

    def get_trade_history(self) -> list[dict[str, Any]]:
        """Get history of filled and closed trades with realized P&L."""
        return []

    def reset_account(self, initial_cash: float | None = None) -> None:
        """Reset account state (paper / simulation mode)."""
        pass
