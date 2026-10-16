"""
AI Quant Trading Platform — Structured Logging

Uses structlog for structured, JSON-capable logging.
Every log entry includes: timestamp, service, level, event, and contextual data.

Why structlog:
- Structured key-value logs (not just strings)
- JSON output for production (machine parseable)
- Console output for development (human readable)
- Context binding (attach request_id, symbol, strategy to all logs in a scope)
- Compatible with standard logging library
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from core.config import Environment, get_settings


def setup_logging(
    level: str | None = None,
    environment: Environment | None = None,
) -> None:
    """
    Configure structured logging for the platform.

    Args:
        level: Log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        environment: Environment override. Development gets colored console output,
                     production gets JSON output.
    """
    settings = get_settings()
    log_level = level or settings.log_level
    env = environment or settings.environment

    # ── Shared processors ───────────────────
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # ── Environment-specific rendering ──────
    if env == Environment.DEVELOPMENT:
        # Pretty colored console output for development
        renderer: Processor = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        # JSON output for production (Loki/ELK compatible)
        renderer = structlog.processors.JSONRenderer()

    # ── Configure structlog ─────────────────
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

    # ── Configure standard logging ──────────
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Reduce noise from third-party libraries
    for noisy_logger in [
        "urllib3",
        "asyncio",
        "websockets",
        "httpx",
        "httpcore",
        "sqlalchemy.engine",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger with an initial context.

    Usage:
        logger = get_logger("market_data", symbol="AAPL")
        logger.info("bar_closed", timeframe="5m", close=150.25)
        
        # Output (dev):
        # 2026-09-08T14:30:00Z [info] bar_closed  symbol=AAPL timeframe=5m close=150.25
        
        # Output (prod JSON):
        # {"timestamp":"2026-09-08T14:30:00Z","level":"info","event":"bar_closed",
        #  "logger":"market_data","symbol":"AAPL","timeframe":"5m","close":150.25}

    Args:
        name: Logger name (usually the module/service name).
        **initial_context: Key-value pairs bound to all log entries from this logger.

    Returns:
        A bound structured logger.
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
