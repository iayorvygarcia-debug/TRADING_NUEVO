"""
Risk engine service package.

Provides deterministic pre-trade risk validation, 5-tier circuit breakers,
and autonomous kill switch.
"""

from services.risk_engine.models import (
    RiskLevel,
    PreTradeCheckResult,
    PortfolioRiskState,
)
from services.risk_engine.kill_switch import (
    KillSwitch,
    get_kill_switch,
)
from services.risk_engine.circuit_breaker import (
    CircuitBreaker,
)
from services.risk_engine.pre_trade_checker import (
    PreTradeChecker,
)
from services.risk_engine.engine import (
    RiskEngine,
    get_risk_engine,
)

__all__ = [
    "RiskLevel",
    "PreTradeCheckResult",
    "PortfolioRiskState",
    "KillSwitch",
    "get_kill_switch",
    "CircuitBreaker",
    "PreTradeChecker",
    "RiskEngine",
    "get_risk_engine",
]
