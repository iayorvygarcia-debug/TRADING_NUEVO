"""
AI Quant Trading Platform — Structured Logging

Uses structlog when available, with a seamless standard-library structured logger
fallback when structlog is not installed in the active Python environment.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from core.config import Environment, get_settings

try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    structlog = None  # type: ignore
    _HAS_STRUCTLOG = False


class _FallbackBoundLogger:
    """Lightweight structured logger fallback when structlog is not installed."""

    def __init__(self, name: str = "quant", **context: Any) -> None:
        self._logger = logging.getLogger(name)
        self._context = context

    def bind(self, **new_context: Any) -> "_FallbackBoundLogger":
        merged = {**self._context, **new_context}
        return _FallbackBoundLogger(self._logger.name, **merged)

    def _fmt(self, event: str, **kw: Any) -> str:
        merged = {**self._context, **kw}
        kv_str = " ".join(f"{k}={v}" for k, v in merged.items())
        return f"{event} {kv_str}".strip()

    def debug(self, event: str, **kw: Any) -> None:
        self._logger.debug(self._fmt(event, **kw))

    def info(self, event: str, **kw: Any) -> None:
        self._logger.info(self._fmt(event, **kw))

    def warning(self, event: str, **kw: Any) -> None:
        self._logger.warning(self._fmt(event, **kw))

    def error(self, event: str, **kw: Any) -> None:
        self._logger.error(self._fmt(event, **kw))

    def critical(self, event: str, **kw: Any) -> None:
        self._logger.critical(self._fmt(event, **kw))


def setup_logging(
    level: str | None = None,
    environment: Environment | None = None,
) -> None:
    settings = get_settings()
    log_level = level or settings.log_level
    env = environment or settings.environment

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    if _HAS_STRUCTLOG:
        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]
        if env == Environment.DEVELOPMENT:
            renderer = structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        else:
            renderer = structlog.processors.JSONRenderer()

        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.format_exc_info,
                renderer,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    for noisy_logger in [
        "urllib3", "asyncio", "websockets", "httpx", "httpcore", "sqlalchemy.engine"
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str, **initial_context: Any) -> Any:
    if _HAS_STRUCTLOG:
        logger = structlog.get_logger(name)
        if initial_context:
            logger = logger.bind(**initial_context)
        return logger
    return _FallbackBoundLogger(name, **initial_context)
