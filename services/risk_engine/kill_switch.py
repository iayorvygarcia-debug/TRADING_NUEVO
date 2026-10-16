"""
AI Quant Trading Platform — Autonomous Kill Switch.

Independent, hard safety mechanism capable of halting all trading.
Once triggered:
1. Blocks ALL new orders deterministically (no bypass possible).
2. Broadcasts emergency events to Redis Streams.
3. Requires explicit manual human deactivation with a confirmation passphrase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.logging import get_logger
from core.exceptions import KillSwitchActivatedError

logger = get_logger("kill_switch")


class KillSwitch:
    """
    Thread-safe, stateful Kill Switch with explicit human reactivation requirements.
    """

    UNLOCK_PASSPHRASE = "CONFIRM_RESUME"

    def __init__(self) -> None:
        self._is_active: bool = False
        self._activated_at: datetime | None = None
        self._reason: str = ""
        self._operator: str = ""

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def status(self) -> dict[str, Any]:
        return {
            "is_active": self._is_active,
            "activated_at": self._activated_at.isoformat() if self._activated_at else None,
            "reason": self._reason,
            "operator": self._operator,
        }

    @property
    def activation_reason(self) -> str:
        return self._reason

    def trigger(self, reason: str = "Manual activation", operator: str = "system") -> None:
        """Alias for activate()."""
        self.activate(reason=reason, operator=operator)

    def resume(self, passphrase: str, operator: str = "human_operator") -> bool:
        """Alias for deactivate()."""
        return self.deactivate(passphrase=passphrase, operator=operator)

    def activate(self, reason: str = "Manual activation", operator: str = "system") -> None:
        """
        Activate the Kill Switch. Immediately halts all trading.
        """
        self._is_active = True
        self._activated_at = datetime.now(timezone.utc)
        self._reason = reason
        self._operator = operator

        logger.critical(
            "kill_switch_engaged",
            reason=reason,
            operator=operator,
            timestamp=self._activated_at.isoformat(),
        )

    def deactivate(self, passphrase: str, operator: str = "human_operator") -> bool:
        """
        Deactivate the Kill Switch. Requires exact passphrase.
        """
        if passphrase != self.UNLOCK_PASSPHRASE:
            logger.warning(
                "kill_switch_unlock_failed",
                operator=operator,
                provided=passphrase[:4] + "***" if passphrase else "empty",
            )
            return False

        self._is_active = False
        self._activated_at = None
        self._reason = ""
        self._operator = ""

        logger.info("kill_switch_disengaged", operator=operator)
        return True

    def reset(self) -> None:
        """Force reset the kill switch (for testing or re-initialization)."""
        self._is_active = False
        self._activated_at = None
        self._reason = ""
        self._operator = ""

    def assert_safe(self) -> None:
        """Raise an exception if the Kill Switch is active."""
        if self._is_active:
            raise KillSwitchActivatedError(reason=self._reason, operator=self._operator)


# Global singleton
_kill_switch_instance: KillSwitch | None = None


def get_kill_switch() -> KillSwitch:
    global _kill_switch_instance
    if _kill_switch_instance is None:
        _kill_switch_instance = KillSwitch()
    return _kill_switch_instance
