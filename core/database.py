"""
AI Quant Trading Platform — Database Connection Manager

Manages async PostgreSQL connections using SQLAlchemy 2.0 async engine.
Designed for TimescaleDB (100% PostgreSQL compatible).

Why asyncpg + SQLAlchemy async:
- asyncpg: fastest PostgreSQL driver for Python
- SQLAlchemy 2.0: modern async API, migration support via Alembic
- Connection pooling: reuse connections efficiently
- Compatible with TimescaleDB hypertables

How it works:
- Creates an async engine and session factory on startup
- Services acquire sessions from the pool for each operation
- Sessions are automatically returned to the pool
- Health check pings the database periodically
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from core.config import get_settings
from core.logging import get_logger

logger = get_logger("database")

# ── Module-level state ──────────────────────
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_database() -> None:
    """
    Initialize the database engine and session factory.
    
    Call this once at application startup.
    """
    global _engine, _session_factory

    settings = get_settings()
    db_url = settings.database.async_url

    logger.info(
        "initializing_database",
        host=settings.database.host,
        port=settings.database.port,
        database=settings.database.name,
    )

    _engine = create_async_engine(
        db_url,
        echo=settings.debug,
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,  # Recycle connections after 1 hour
        pool_pre_ping=True,  # Verify connections before use
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Verify connection
    try:
        async with _engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.scalar()
        logger.info("database_connected")
    except Exception as e:
        logger.error("database_connection_failed", error=str(e))
        raise


async def close_database() -> None:
    """Close database engine and all connections. Call at shutdown."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        logger.info("database_closed")
        _engine = None
        _session_factory = None


def get_engine() -> AsyncEngine:
    """Get the async database engine. Raises if not initialized."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session from the pool.
    
    Usage:
        async with get_session() as session:
            result = await session.execute(text("SELECT * FROM instruments"))
            rows = result.fetchall()
    
    Sessions are automatically committed on success and rolled back on error.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def health_check() -> dict:
    """
    Check database health.
    
    Returns:
        Dictionary with health status and latency.
    """
    import time

    try:
        start = time.monotonic()
        async with get_session() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
        latency_ms = (time.monotonic() - start) * 1000

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
