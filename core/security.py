"""
AI Quant Trading Platform — Security Module

Handles:
- Secrets management (environment variables only, never in code)
- API key validation
- Sensitive data masking in logs

Security rules enforced by this module:
1. API keys are NEVER printed in logs
2. API keys are NEVER stored in source code
3. All secrets come from environment variables or .env file
4. Broker API keys are validated before any trading operation
"""

from __future__ import annotations

import os
import re
from typing import Any

from core.config import get_settings
from core.logging import get_logger

logger = get_logger("security")

# ── Sensitive patterns to mask in logs ──────
SENSITIVE_PATTERNS = [
    re.compile(r"(api[_-]?key|secret[_-]?key|password|token|authorization)\s*[:=]\s*\S+", re.I),
    re.compile(r"(ALPACA_API_KEY|ALPACA_SECRET_KEY|FRED_API_KEY|FINNHUB_API_KEY)\s*=\s*\S+", re.I),
]


def mask_sensitive(text: str) -> str:
    """
    Mask sensitive values in text (for safe logging).
    
    Replaces API keys, passwords, tokens with asterisks.
    """
    masked = text
    for pattern in SENSITIVE_PATTERNS:
        masked = pattern.sub(lambda m: m.group().split("=")[0] + "=***MASKED***", masked)
    return masked


def validate_api_keys() -> dict[str, bool]:
    """
    Validate that required API keys are present.
    
    Returns:
        Dictionary of service → key_present status.
    """
    settings = get_settings()

    status = {
        "alpaca": settings.alpaca.validate_keys(),
        "sec_edgar": bool(settings.sec.user_agent),  # SEC doesn't need API key
        "fred": bool(settings.fred.api_key),
        "telegram": settings.notifications.telegram_enabled,
    }

    # Log status (without revealing actual keys)
    for service, has_key in status.items():
        if has_key:
            logger.info("api_key_present", service=service)
        else:
            logger.warning("api_key_missing", service=service)

    return status


def require_api_key(service: str) -> str:
    """
    Get an API key or raise an error if not configured.
    
    Args:
        service: Service name (alpaca, fred, finnhub, etc.)
    
    Returns:
        The API key string.
    
    Raises:
        ValueError if the key is not configured.
    """
    settings = get_settings()

    key_map: dict[str, str] = {
        "alpaca_key": settings.alpaca.api_key,
        "alpaca_secret": settings.alpaca.secret_key,
        "fred": settings.fred.api_key,
    }

    key = key_map.get(service, "")
    if not key:
        raise ValueError(
            f"API key for '{service}' is not configured. "
            f"Set the corresponding environment variable in .env"
        )

    return key


def safe_dict(data: dict[str, Any], mask_keys: set[str] | None = None) -> dict[str, Any]:
    """
    Create a copy of a dict with sensitive keys masked.
    
    Args:
        data: Original dictionary.
        mask_keys: Set of key names to mask. Defaults to common sensitive keys.
    
    Returns:
        New dictionary with sensitive values replaced by "***".
    """
    if mask_keys is None:
        mask_keys = {
            "api_key", "secret_key", "password", "token",
            "authorization", "secret", "credentials",
        }

    safe = {}
    for k, v in data.items():
        if k.lower() in mask_keys:
            safe[k] = "***MASKED***"
        elif isinstance(v, dict):
            safe[k] = safe_dict(v, mask_keys)
        else:
            safe[k] = v

    return safe
