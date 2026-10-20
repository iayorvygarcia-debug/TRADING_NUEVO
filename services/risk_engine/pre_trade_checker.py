"""AI Quant Trading Platform — Pre-Trade Risk Checker."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone

from core.config import get_settings
from core.logging import get_logger
from core.market_session import can_trade
from services.risk_engine.models import PortfolioRiskState, PreTradeCheckResult
from services.risk_engine.circuit_breaker import CircuitBreaker
from services.risk_engine.kill_switch import KillSwitch, get_kill_switch

logger = get_logger("pre_trade_checker")


class PreTradeChecker:
    """Deterministic pre-trade risk filter."""

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.settings = get_settings().risk
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.kill_switch = kill_switch or get_kill_switch()
        self._order_timestamps: deque[float] = deque()

    def check(
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
        violations: list[str] = []
        reasons: list[str] = []

        notional = quantity * price
        equity = max(1.0, portfolio_state.equity)

        if self.kill_switch.is_active:
            violations.append(f"KILL SWITCH ACTIVE: {self.kill_switch.status.get('reason')}")
            return PreTradeCheckResult(
                approved=False,
                symbol=symbol,
                direction=direction,
                requested_quantity=quantity,
                approved_quantity=0.0,
                violations=violations,
                reasons=["Rejected by hard Kill Switch."],
                circuit_breaker_level=self.circuit_breaker.level,
            )

        cb_level = self.circuit_breaker.evaluate(portfolio_state)
        if not is_exit and not self.circuit_breaker.can_open_new_positions:
            violations.append(f"Circuit Breaker in {cb_level.name} state (new positions blocked)")

        now_utc = datetime.now(timezone.utc)
        if not can_trade(now_utc, allow_premarket=extended_hours or is_exit, allow_afterhours=extended_hours or is_exit):
            violations.append("Market is closed for regular trading hours.")

        if not is_exit and portfolio_state.daily_pnl_pct <= -(self.settings.max_daily_loss * 100.0):
            violations.append(
                f"Max daily loss exceeded: {portfolio_state.daily_pnl_pct:.2f}% "
                f"<= -{self.settings.max_daily_loss * 100.0:.2f}%"
            )

        if not is_exit and portfolio_state.drawdown_pct >= (self.settings.max_drawdown * 100.0):
            violations.append(
                f"Max drawdown limit exceeded: {portfolio_state.drawdown_pct:.2f}% "
                f">= {self.settings.max_drawdown * 100.0:.2f}%"
            )

        if not is_exit and portfolio_state.daily_trades_count >= self.settings.max_trades_per_day:
            violations.append(
                f"Max daily trades limit reached: {portfolio_state.daily_trades_count} "
                f">= {self.settings.max_trades_per_day}"
            )

        now_mono = time.monotonic()
        while self._order_timestamps and (now_mono - self._order_timestamps[0] > 60.0):
            self._order_timestamps.popleft()

        if len(self._order_timestamps) >= self.settings.max_orders_per_minute:
            violations.append(
                f"Order submission rate limit exceeded: {len(self._order_timestamps)} "
                f"orders in last 60s (max {self.settings.max_orders_per_minute}/min)"
            )

        from core.market_session import get_dynamic_max_spread
        session_tag = "premarket" if extended_hours else "regular"
        dyn_spread = get_dynamic_max_spread(symbol=symbol, session=session_tag, natr_14=natr_14)
        effective_max = max(self.settings.max_spread_pct, dyn_spread) if getattr(self.settings, "max_spread_pct", None) else dyn_spread
        if current_spread_pct > effective_max:
            violations.append(
                f"Spread too wide: {current_spread_pct * 100.0:.3f}% "
                f"> {effective_max * 100.0:.3f}% (illiquid asset)"
            )

        if not is_exit and signal_score < self.settings.min_signal_score:
            violations.append(
                f"Signal composite score ({signal_score:.2f}) is below minimum "
                f"threshold ({self.settings.min_signal_score:.2f})"
            )

        current_sector_exp = portfolio_state.sector_exposures.get(sector, 0.0)
        projected_sector_exp = current_sector_exp + (notional / equity)
        if not is_exit and projected_sector_exp > self.settings.max_sector_exposure:
            violations.append(
                f"Sector '{sector}' exposure would reach {projected_sector_exp * 100.0:.1f}% "
                f"(max {self.settings.max_sector_exposure * 100.0:.1f}%)"
            )

        max_allowed_notional = equity * self.settings.max_position_size
        approved_qty = quantity

        if not is_exit and notional > max_allowed_notional:
            approved_qty = max_allowed_notional / price
            reasons.append(
                f"Position size scaled down from {quantity:.1f} to {approved_qty:.1f} shares "
                f"to respect max {self.settings.max_position_size * 100.0:.1f}% equity limit."
            )

        multiplier = self.circuit_breaker.position_size_multiplier
        if not is_exit and multiplier < 1.0 and approved_qty > 0:
            approved_qty *= multiplier
            reasons.append(
                f"Position size reduced by {round((1.0 - multiplier) * 100)}% "
                f"due to Circuit Breaker {cb_level.name} tier."
            )

        approved_qty = max(0.0, float(int(approved_qty)))
        if approved_qty < 1.0 and not violations:
            violations.append("Calculated share quantity is below 1 share minimum.")

        is_approved = len(violations) == 0

        if is_approved:
            self._order_timestamps.append(now_mono)
            reasons.append("All pre-trade deterministic safety gates passed successfully.")

        return PreTradeCheckResult(
            approved=is_approved,
            symbol=symbol,
            direction=direction,
            requested_quantity=quantity,
            approved_quantity=approved_qty,
            reasons=reasons,
            violations=violations,
            circuit_breaker_level=cb_level,
        )
