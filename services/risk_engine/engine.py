"""
AI Quant Trading Platform — Central Risk Engine.

Master safety coordinator combining:
- Pre-Trade Gatekeeper (Deterministic risk checks)
- 5-Tier Circuit Breaker (Autonomous risk escalation)
- Hard Kill Switch (Emergency immediate trading halt)
- Event Bus audit publishing
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.logging import get_logger
from core import event_bus
from services.risk_engine.models import PortfolioRiskState, PreTradeCheckResult, RiskLevel
from services.risk_engine.circuit_breaker import CircuitBreaker
from services.risk_engine.kill_switch import KillSwitch, get_kill_switch
from services.risk_engine.pre_trade_checker import PreTradeChecker

logger = get_logger("risk_engine")


class RiskEngine:
    """
    Central Risk Management System.
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.kill_switch = kill_switch or get_kill_switch()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.pre_trade_checker = PreTradeChecker(
            circuit_breaker=self.circuit_breaker,
            kill_switch=self.kill_switch,
        )
        self._session_date: str | None = None
        self._daily_starting_equity: float = 0.0
        self._daily_peak_equity: float = 0.0

    def get_or_create_portfolio_state(
        self,
        equity: float,
        cash: float,
        open_positions_count: int = 0,
        consecutive_losses: int = 0,
        sector_exposures: dict[str, float] | None = None,
        daily_starting_equity: float | None = None,
        peak_equity: float | None = None,
    ) -> PortfolioRiskState:
        """
        Builds or updates session-aware PortfolioRiskState with daily peak tracking.
        Resets automatically when a new market date (Eastern Time) begins.
        """
        from datetime import datetime
        from core.market_session import ET

        now_et = datetime.now(ET)
        session_date = now_et.date().isoformat()

        if self._session_date != session_date:
            self._session_date = session_date
            self._daily_starting_equity = daily_starting_equity if (daily_starting_equity and daily_starting_equity > 0) else equity
            self._daily_peak_equity = max(self._daily_starting_equity, equity, peak_equity or 0.0)
        else:
            if daily_starting_equity and daily_starting_equity > 0:
                self._daily_starting_equity = daily_starting_equity
            if peak_equity and peak_equity > 0:
                self._daily_peak_equity = max(self._daily_peak_equity, peak_equity)
            self._daily_peak_equity = max(self._daily_peak_equity, equity)

        return PortfolioRiskState(
            equity=equity,
            peak_equity=self._daily_peak_equity,
            cash=cash,
            daily_starting_equity=self._daily_starting_equity,
            open_positions_count=open_positions_count,
            consecutive_losses=consecutive_losses,
            sector_exposures=sector_exposures or {},
        )

    def evaluate_order(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        price: float,
        portfolio_state: PortfolioRiskState,
        signal_score: float = 1.0,
        current_spread_pct: float = 0.001,
        sector: str = "technology",
        is_exit: bool = False,
        extended_hours: bool = False,
        natr_14: float | None = None,
    ) -> PreTradeCheckResult:
        """
        Validate order against all pre-trade deterministic safety gates.
        """
        result = self.pre_trade_checker.check(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            portfolio_state=portfolio_state,
            signal_score=signal_score,
            current_spread_pct=current_spread_pct,
            sector=sector,
            is_exit=is_exit,
            extended_hours=extended_hours,
            natr_14=natr_14,
        )

        return result


    def update_portfolio_state(self, state: PortfolioRiskState) -> RiskLevel:
        """
        Feed new portfolio metrics into the Circuit Breaker.
        """
        return self.circuit_breaker.evaluate(state)

    def trigger_kill_switch(self, reason: str, operator: str = "risk_engine") -> None:
        """Emergency engage of the Kill Switch."""
        self.kill_switch.activate(reason=reason, operator=operator)

    def resume_trading(self, passphrase: str, operator: str = "human_admin") -> bool:
        """Disengage Kill Switch and reset circuit breaker if conditions allow."""
        unlocked = self.kill_switch.deactivate(passphrase=passphrase, operator=operator)
        if unlocked:
            self.circuit_breaker.reset()
            return True
        return False

    def get_status(self) -> dict[str, Any]:
        """Comprehensive risk health summary for API and dashboards."""
        return {
            "kill_switch": self.kill_switch.status,
            "circuit_breaker": self.circuit_breaker.get_status(),
        }


# Singleton accessor
_risk_engine_instance: RiskEngine | None = None


def get_risk_engine() -> RiskEngine:
    global _risk_engine_instance
    if _risk_engine_instance is None:
        _risk_engine_instance = RiskEngine()
    return _risk_engine_instance
