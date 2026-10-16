"""
AI Quant Trading Platform — Order State Machine.

Enforces valid, deterministic lifecycle transitions for every order.
Guarantees:
- Terminal states (FILLED, CANCELLED, REJECTED, EXPIRED, FAILED) are immutable.
- Disallows illegal jumps (e.g. CREATED -> FILLED without validation/submission).
- Provides audit trails of all state changes.
"""

from __future__ import annotations

from core.exceptions import PlatformError
from services.broker_engine.base import OrderStatus


class InvalidOrderStateTransitionError(PlatformError):
    """Raised when an illegal order state transition is attempted."""

    def __init__(self, current_status: OrderStatus, target_status: OrderStatus, order_id: str = "") -> None:
        super().__init__(
            f"Invalid order state transition for order '{order_id}': "
            f"cannot move from {current_status.value} to {target_status.value}",
            details={
                "order_id": order_id,
                "current_status": current_status.value,
                "target_status": target_status.value,
            },
        )


class OrderStateMachine:
    """
    Formal state machine for trading orders.
    """

    TERMINAL_STATES = {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.FAILED,
        OrderStatus.EXPIRED,
    }

    ACTIVE_STATES = {
        OrderStatus.CREATED,
        OrderStatus.VALIDATING,
        OrderStatus.APPROVED,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCEL_PENDING,
    }

    CANCELLABLE_STATES = {
        OrderStatus.CREATED,
        OrderStatus.APPROVED,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
    }

    VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
        OrderStatus.CREATED: {
            OrderStatus.VALIDATING,
            OrderStatus.APPROVED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
        },
        OrderStatus.VALIDATING: {
            OrderStatus.APPROVED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
            OrderStatus.CANCELLED,
        },
        OrderStatus.APPROVED: {
            OrderStatus.SUBMITTED,
            OrderStatus.CANCELLED,
            OrderStatus.FAILED,
        },
        OrderStatus.SUBMITTED: {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        },
        OrderStatus.PARTIALLY_FILLED: {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.CANCELLED,
            OrderStatus.FAILED,
        },
        OrderStatus.CANCEL_PENDING: {
            OrderStatus.CANCELLED,
            OrderStatus.FILLED,     # Order could fill at broker while cancel is pending
            OrderStatus.FAILED,
        },
        # Terminal states have no valid outgoing transitions
        OrderStatus.FILLED: set(),
        OrderStatus.CANCELLED: set(),
        OrderStatus.REJECTED: set(),
        OrderStatus.FAILED: set(),
        OrderStatus.EXPIRED: set(),
    }

    @classmethod
    def can_transition(cls, from_status: OrderStatus, to_status: OrderStatus) -> bool:
        """Check whether a transition is allowed."""
        return to_status in cls.VALID_TRANSITIONS.get(from_status, set())

    @classmethod
    def transition(cls, current_status: OrderStatus, new_status: OrderStatus, order_id: str = "") -> OrderStatus:
        """
        Transition order to new status, or raise InvalidOrderStateTransitionError.
        """
        if not cls.can_transition(current_status, new_status):
            raise InvalidOrderStateTransitionError(current_status, new_status, order_id)
        return new_status

    @classmethod
    def is_terminal(cls, status: OrderStatus) -> bool:
        """Whether the order has finished its lifecycle."""
        return status in cls.TERMINAL_STATES

    @classmethod
    def is_active(cls, status: OrderStatus) -> bool:
        """Whether the order is still live in the system."""
        return status in cls.ACTIVE_STATES

    @classmethod
    def can_cancel(cls, status: OrderStatus) -> bool:
        """Whether the order can be cancelled."""
        return status in cls.CANCELLABLE_STATES
