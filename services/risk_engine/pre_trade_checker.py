"""
AI Quant Trading Platform — Pre-Trade Risk Checker.

Deterministic gatekeeper that validates every proposed order before submission.
No trade can bypass this engine under any circumstance.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from core.config import get_settings
from core.logging import get_logger
from core.market_session import can_trade
from services.risk_engine.models import PortfolioRiskState, PreTradeCheckResult, RiskLevel
from services.risk_engine.circuit_breaker import CircuitBreaker
from services.risk_engine.kill_switch import KillSwitch, get_kill_switch

logger = get_logger("pre_trade_checker")


class PreTradeChecker:
    """
    Deterministic pre-trade risk filter.
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.settings = get_settings().risk
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.kill_switch = kill_switch or get_kill_switch()

        # Rate limiting: timestamps of orders submitted in the last 60s
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
        """
        Run all deterministic risk checks on an incoming proposed trade.
        """
        violations: list[str] = []
        reasons: list[str] = []

        notional = quantity * price
        equity = max(1.0, portfolio_state.equity)

        # ── 1. Kill Switch Check ─────────────────────────────────────
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

        # ── 2. Circuit Breaker Check ─────────────────────────────────
        cb_level = self.circuit_breaker.evaluate(portfolio_state)
        if not is_exit and not self.circuit_breaker.can_open_new_positions:
            violations.append(f"Circuit Breaker in {cb_level.name} state (new positions blocked)")

        # ── 3. Market Session Check ──────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        if not can_trade(now_utc, allow_premarket=extended_hours or is_exit, allow_afterhours=extended_hours or is_exit):
            violations.append("Market is closed for regular trading hours.")

        # ── 4. Max Daily Loss Check ──────────────────────────────────
        if portfolio_state.daily_pnl_pct <= -(self.settings.max_daily_loss * 100.0):
            violations.append(
                f"Max daily loss exceeded: {portfolio_state.daily_pnl_pct:.2f}% "
                f"<= -{self.settings.max_daily_loss * 100.0:.2f}%"
            )

        # ── 5. Max Drawdown Check ────────────────────────────────────
        if portfolio_state.drawdown_pct >= (self.settings.max_drawdown * 100.0):
            violations.append(
                f"Max drawdown limit exceeded: {portfolio_state.drawdown_pct:.2f}% "
                f">= {self.settings.max_drawdown * 100.0:.2f}%"
            )

        # ── 6. Maximum Daily Trades Check ────────────────────────────
        if portfolio_state.daily_trades_count >= self.settings.max_trades_per_day:
            violations.append(
                f"Max daily trades limit reached: {portfolio_state.daily_trades_count} "
                f">= {self.settings.max_trades_per_day}"
            )

        # ── 7. Order Submission Rate Limiting ────────────────────────
        now_mono = time.monotonic()
        while self._order_timestamps and (now_mono - self._order_timestamps[0] > 60.0):
            self._order_timestamps.popleft()

        if len(self._order_timestamps) >= self.settings.max_orders_per_minute:
            violations.append(
                f"Order submission rate limit exceeded: {len(self._order_timestamps)} "
                f"orders in last 60s (max {self.settings.max_orders_per_minute}/min)"
            )

        # ── 8. Bid-Ask Spread Friction Check (Dynamic Model) ─────────
        from core.market_session import get_dynamic_max_spread
        session_tag = "premarket" if extended_hours else "regular"
        dyn_spread = get_dynamic_max_spread(symbol=symbol, session=session_tag, natr_14=natr_14)
        effective_max = max(self.settings.max_spread_pct, dyn_spread) if getattr(self.settings, "max_spread_pct", None) else dyn_spread
        if current_spread_pct > effective_max:
            violations.append(
                f"Spread too wide: {current_spread_pct * 100.0:.3f}% "
                f"> {effective_max * 100.0:.3f}% (illiquid asset)"
            )

        # ── 9. Minimum Signal Score ──────────────────────────────────
        if signal_score < self.settings.min_signal_score:
            violations.append(
                f"Signal composite score ({signal_score:.2f}) is below minimum "
                f"threshold ({self.settings.min_signal_score:.2f})"
            )

        # ── 10. Max Sector Exposure Check ────────────────────────────
        current_sector_exp = portfolio_state.sector_exposures.get(sector, 0.0)
        projected_sector_exp = current_sector_exp + (notional / equity)
        if projected_sector_exp > self.settings.max_sector_exposure:
            violations.append(
                f"Sector '{sector}' exposure would reach {projected_sector_exp * 100.0:.1f}% "
                f"(max {self.settings.max_sector_exposure * 100.0:.1f}%)"
            )

        # ── 11. Max Position Size Sizing / Capping ───────────────────
        max_allowed_notional = equity * self.settings.max_position_size
        approved_qty = quantity

        if notional > max_allowed_notional:
            # Scale down to maximum allowed rather than flat rejection
            approved_qty = max_allowed_notional / price
            reasons.append(
                f"Position size scaled down from {quantity:.1f} to {approved_qty:.1f} shares "
                f"to respect max {self.settings.max_position_size * 100.0:.1f}% equity limit."
            )

        # ── 12. Circuit Breaker Sizing Adjustment ────────────────────
        multiplier = self.circuit_breaker.position_size_multiplier
        if multiplier < 1.0 and approved_qty > 0:
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
            reasons.append(f"All {11} pre-trade deterministic safety gates passed successfully.")
            logger.info(
                "pre_trade_risk_approved",
                symbol=symbol,
                quantity=approved_qty,
                price=price,
                notional=round(approved_qty * price, 2),
            )
        else:
            logger.warning(
                "pre_trade_risk_rejected",
                symbol=symbol,
                violations=violations,
            )

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
