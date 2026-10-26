"""
AI Quant Trading Platform — Multi-Tiered Circuit Breaker.

Continuously monitors portfolio health (Daily PnL, Drawdown, Consecutive Losses).
Enforces 5 operational levels:
- NORMAL (0): All operations permitted.
- WARNING (1): Alerts emitted, sizing reduced to 75%.
- STOP_NEW (2): Halts opening any new positions.
- DE_RISK (3): Cuts portfolio exposure by 50%.
- EMERGENCY (4): Auto-engages the Kill Switch.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from core.logging import get_logger
from services.risk_engine.models import PortfolioRiskState, RiskLevel
from services.risk_engine.kill_switch import get_kill_switch

logger = get_logger("circuit_breaker")


class CircuitBreaker:
    """
    Automated risk circuit breaker with state hysteresis.
    """

    def __init__(
        self,
        warning_daily_loss_pct: float = 1.5,
        stop_new_daily_loss_pct: float = 3.0,
        derisk_daily_loss_pct: float = 4.5,
        emergency_daily_loss_pct: float = 6.0,
        warning_drawdown_pct: float = 5.0,
        stop_new_drawdown_pct: float = 8.0,
        derisk_drawdown_pct: float = 10.0,
        emergency_drawdown_pct: float = 15.0,
        max_consecutive_losses: int = 4,
        cooldown_minutes: int = 30,
        # ── Capa 2: Trailing de Ganancia Diaria (Profit Lock-in) ──────
        min_profit_lock_usd: float = 500.0,
        max_profit_giveback_pct: float = 35.0,
    ) -> None:
        self.warning_loss = warning_daily_loss_pct
        self.stop_new_loss = stop_new_daily_loss_pct
        self.derisk_loss = derisk_daily_loss_pct
        self.emergency_loss = emergency_daily_loss_pct

        self.warning_dd = warning_drawdown_pct
        self.stop_new_dd = stop_new_drawdown_pct
        self.derisk_dd = derisk_drawdown_pct
        self.emergency_dd = emergency_drawdown_pct

        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_period = timedelta(minutes=cooldown_minutes)

        self.min_profit_lock_usd = min_profit_lock_usd
        self.max_profit_giveback_pct = max_profit_giveback_pct

        self._current_level = RiskLevel.NORMAL
        self._last_escalation_time: datetime | None = None
        self._trigger_reasons: list[str] = []

    @property
    def level(self) -> RiskLevel:
        return self._current_level

    @property
    def can_open_new_positions(self) -> bool:
        return self._current_level < RiskLevel.STOP_NEW

    @property
    def position_size_multiplier(self) -> float:
        """Dynamic multiplier applied to trade quantities based on risk tier."""
        if self._current_level == RiskLevel.NORMAL:
            return 1.0
        elif self._current_level == RiskLevel.WARNING:
            return 0.75  # Scale down by 25%
        elif self._current_level == RiskLevel.DE_RISK:
            return 0.50  # Scale down by 50%
        return 0.0

    def evaluate(self, state: PortfolioRiskState) -> RiskLevel:
        """
        Evaluate current portfolio metrics and adjust circuit breaker level.
        Applies Layer 2 (Daily Profit Lock-in) and Layer 3 (Capital Circuit Breaker).
        """
        daily_loss_pct = abs(min(0.0, state.daily_pnl_pct))
        drawdown_pct = state.drawdown_pct
        losses = state.consecutive_losses

        # Capa 2: Trailing de Ganancia Diaria (Profit Lock-in)
        # Solo se activa si hubo ganancia significativa acumulada (>= min_profit_lock_usd)
        # Y si la cuenta todavía está en ganancia (daily_pnl > 0).
        # Si la cuenta está en pérdida, el riesgo es controlado por límites diarios de pérdida, no por profit-lock.
        profit_lock_breached = bool(
            state.daily_peak_profit >= self.min_profit_lock_usd
            and state.daily_pnl > 0
            and state.profit_giveback_pct >= self.max_profit_giveback_pct
        )

        reasons: list[str] = []
        new_level = RiskLevel.NORMAL

        # ── Level 4: EMERGENCY ───────────────────────────────────────
        if daily_loss_pct >= self.emergency_loss or drawdown_pct >= self.emergency_dd:
            new_level = RiskLevel.EMERGENCY
            reasons.append(
                f"Emergency threshold breached: Daily Loss {daily_loss_pct:.2f}% (max {self.emergency_loss}%) "
                f"or Drawdown {drawdown_pct:.2f}% (max {self.emergency_dd}%)"
            )
            # Automatically engage Kill Switch
            kill_switch = get_kill_switch()
            if not kill_switch.is_active:
                kill_switch.activate(
                    reason=f"Circuit Breaker L4 Triggered: {reasons[0]}",
                    operator="circuit_breaker_l4",
                )

        # ── Level 3: DE_RISK ─────────────────────────────────────────
        elif daily_loss_pct >= self.derisk_loss or drawdown_pct >= self.derisk_dd:
            new_level = RiskLevel.DE_RISK
            reasons.append(
                f"De-risk threshold breached: Daily Loss {daily_loss_pct:.2f}% "
                f"or Drawdown {drawdown_pct:.2f}%"
            )

        # ── Level 2: STOP_NEW ────────────────────────────────────────
        elif (
            daily_loss_pct >= self.stop_new_loss or
            drawdown_pct >= self.stop_new_dd or
            losses >= self.max_consecutive_losses or
            profit_lock_breached
        ):
            new_level = RiskLevel.STOP_NEW
            if profit_lock_breached:
                reasons.append(
                    f"Capa 2 (Profit Lock) activada: Ganancia pico hoy ${state.daily_peak_profit:.2f}, "
                    f"devolvió {state.profit_giveback_pct:.1f}% (máx {self.max_profit_giveback_pct}%). "
                    f"Bloqueando nuevas compras para proteger las ganancias acumuladas."
                )
            if daily_loss_pct >= self.stop_new_loss or drawdown_pct >= self.stop_new_dd or losses >= self.max_consecutive_losses:
                reasons.append(
                    f"Stop-new threshold breached: Daily Loss {daily_loss_pct:.2f}%, "
                    f"Drawdown {drawdown_pct:.2f}%, Consecutive Losses {losses}"
                )

        # ── Level 1: WARNING ─────────────────────────────────────────
        elif daily_loss_pct >= self.warning_loss or drawdown_pct >= self.warning_dd:
            new_level = RiskLevel.WARNING
            reasons.append(
                f"Warning threshold reached: Daily Loss {daily_loss_pct:.2f}% "
                f"or Drawdown {drawdown_pct:.2f}%"
            )

        # Update state and handle escalation
        if new_level != self._current_level:
            old_level = self._current_level
            self._current_level = new_level
            self._trigger_reasons = reasons
            now = datetime.now(timezone.utc)

            if new_level > old_level:
                self._last_escalation_time = now
                logger.warning(
                    "circuit_breaker_escalated",
                    from_level=old_level.name,
                    to_level=new_level.name,
                    reasons=reasons,
                )
            else:
                logger.info(
                    "circuit_breaker_de_escalated",
                    from_level=old_level.name,
                    to_level=new_level.name,
                )

        return self._current_level

    def reset(self) -> None:
        """Reset circuit breaker back to NORMAL."""
        self._current_level = RiskLevel.NORMAL
        self._last_escalation_time = None
        self._trigger_reasons.clear()
        logger.info("circuit_breaker_reset")

    def get_status(self) -> dict[str, Any]:
        return {
            "level": self._current_level.value,
            "level_name": self._current_level.name,
            "can_open_new_positions": self.can_open_new_positions,
            "size_multiplier": self.position_size_multiplier,
            "min_profit_lock_usd": self.min_profit_lock_usd,
            "max_profit_giveback_pct": self.max_profit_giveback_pct,
            "reasons": self._trigger_reasons,
        }


_circuit_breaker_instance: CircuitBreaker | None = None


def get_circuit_breaker() -> CircuitBreaker:
    """Returns the singleton CircuitBreaker instance."""
    global _circuit_breaker_instance
    if _circuit_breaker_instance is None:
        try:
            from core.config import get_settings
            risk = get_settings().risk
            _circuit_breaker_instance = CircuitBreaker(
                min_profit_lock_usd=float(risk.min_profit_lock_usd),
                max_profit_giveback_pct=float(risk.max_profit_giveback_pct),
            )
        except Exception:
            _circuit_breaker_instance = CircuitBreaker()
    return _circuit_breaker_instance

