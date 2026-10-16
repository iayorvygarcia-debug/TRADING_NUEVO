"""
AI Quant Trading Platform — Custom Exceptions

Hierarchical exception system for clear error handling across all services.
Every exception carries context about what happened and why.

Why custom exceptions:
- Each service can raise domain-specific errors
- Circuit breakers and kill switch need specific exception types
- Audit log needs to distinguish error categories
- Error handling can be granular (catch DataError vs BrokerError)
"""

from __future__ import annotations

from typing import Any


class PlatformError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# ── Data Layer Errors ───────────────────────

class DataError(PlatformError):
    """Base for all data-related errors."""


class DataQualityError(DataError):
    """Data failed quality validation (stale, corrupted, anomalous)."""


class DataFeedDisconnectedError(DataError):
    """Market data feed lost connection."""


class StaleDataError(DataError):
    """Data is older than acceptable threshold."""


class MissingDataError(DataError):
    """Expected data is not available."""


# ── Broker Errors ───────────────────────────

class BrokerError(PlatformError):
    """Base for all broker-related errors."""


class BrokerConnectionError(BrokerError):
    """Cannot connect to broker."""


class BrokerStateUnknownError(BrokerError):
    """Raised when broker state (account, positions, open orders) cannot be verified. Fail-safe halts new orders."""


class OrderRejectedError(BrokerError):
    """Broker rejected the order."""

    def __init__(
        self, message: str, order_id: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, details={"order_id": order_id, **kwargs})
        self.order_id = order_id


class OrderTimeoutError(BrokerError):
    """Order submission or fill timed out."""


class DuplicateOrderError(BrokerError):
    """Attempted to submit a duplicate order."""


class InsufficientFundsError(BrokerError):
    """Not enough buying power."""


class ReconciliationError(BrokerError):
    """Local state doesn't match broker state."""


# ── Risk Errors ─────────────────────────────

class RiskError(PlatformError):
    """Base for all risk-related errors."""


class RiskLimitExceededError(RiskError):
    """A risk limit has been exceeded."""

    def __init__(self, limit_name: str, current: float, maximum: float) -> None:
        super().__init__(
            f"Risk limit '{limit_name}' exceeded: {current:.4f} > {maximum:.4f}",
            details={
                "limit_name": limit_name,
                "current_value": current,
                "max_value": maximum,
            },
        )


class CircuitBreakerTriggeredError(RiskError):
    """Circuit breaker has been triggered."""

    def __init__(self, level: int, reason: str) -> None:
        super().__init__(
            f"Circuit breaker L{level} triggered: {reason}",
            details={"level": level, "reason": reason},
        )


class KillSwitchActivatedError(RiskError):
    """Kill switch has been activated — all trading halted."""

    def __init__(self, reason: str, operator: str = "system") -> None:
        super().__init__(
            f"🛑 KILL SWITCH ACTIVATED by {operator}: {reason}",
            details={"reason": reason, "operator": operator},
        )


# ── Signal / Strategy Errors ────────────────

class SignalError(PlatformError):
    """Base for signal/strategy errors."""


class NoTradeDecision(SignalError):
    """
    Not an error — NO TRADE is a valid decision.
    Used to explicitly indicate the system chose not to trade.
    """

    def __init__(self, symbol: str, reason: str) -> None:
        super().__init__(
            f"NO TRADE for {symbol}: {reason}",
            details={"symbol": symbol, "reason": reason},
        )


# ── ML Errors ───────────────────────────────

class MLError(PlatformError):
    """Base for ML-related errors."""


class ModelNotFoundError(MLError):
    """Requested model version not found in registry."""


class ModelDriftError(MLError):
    """Model performance has degraded beyond threshold."""


# ── Infrastructure Errors ───────────────────

class InfrastructureError(PlatformError):
    """Base for infrastructure errors (DB, Redis, network)."""


class DatabaseError(InfrastructureError):
    """Database connection or query error."""


class RedisError(InfrastructureError):
    """Redis connection or operation error."""


class ServiceHealthError(InfrastructureError):
    """A monitored service is unhealthy."""

    def __init__(self, service: str, status: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"Service '{service}' is {status}",
            details={"service": service, "status": status, **(details or {})},
        )


# ── Configuration Errors ────────────────────

class ConfigurationError(PlatformError):
    """Invalid or missing configuration."""


class LiveTradingNotAllowedError(ConfigurationError):
    """Attempted live trading without completing safety checklist."""

    def __init__(self, missing_checks: list[str]) -> None:
        super().__init__(
            f"Live trading not allowed. Missing checks: {', '.join(missing_checks)}",
            details={"missing_checks": missing_checks},
        )
