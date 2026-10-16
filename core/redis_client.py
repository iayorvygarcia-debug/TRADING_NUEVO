"""
AI Quant Trading Platform — Redis Client

Manages Redis connections for:
1. Caching (market data, feature store, session data)
2. Event Bus (Redis Streams for inter-service communication)
3. Pub/Sub (real-time notifications)
4. Rate limiting (API call tracking)

Why Redis:
- Sub-millisecond latency for cache hits
- Redis Streams provide persistent, ordered event log
- Consumer groups for parallel processing
- Built-in TTL for automatic cache expiration
- Simpler than Kafka for MVP volume (~1000 events/s)
"""

from __future__ import annotations

from typing import Any

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore

from core.config import get_settings
from core.logging import get_logger

logger = get_logger("redis")

# ── Module-level state ──────────────────────
_redis: aioredis.Redis | None = None


async def init_redis() -> None:
    """
    Initialize Redis connection pool.
    Call once at application startup.
    """
    global _redis

    settings = get_settings()

    logger.info(
        "initializing_redis",
        host=settings.redis.host,
        port=settings.redis.port,
        db=settings.redis.db,
    )

    _redis = aioredis.Redis(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password or None,
        db=settings.redis.db,
        decode_responses=True,
        max_connections=50,
        socket_timeout=1.0,
        socket_connect_timeout=1.0,
    )

    # Verify connection
    try:
        import asyncio
        await asyncio.wait_for(_redis.ping(), timeout=1.0)
        logger.info("redis_connected")
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        _redis = None
        raise


def is_redis_available() -> bool:
    """Check if Redis client is initialized and active."""
    return _redis is not None


async def close_redis() -> None:
    """Close Redis connection. Call at shutdown."""
    global _redis

    if _redis is not None:
        await _redis.aclose()
        logger.info("redis_closed")
        _redis = None


def get_redis() -> Any:
    """Get the Redis client. Raises if not initialized."""
    if _redis is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis


# ── Cache Helpers ───────────────────────────

async def cache_set(
    key: str,
    value: str,
    ttl_seconds: int = 300,
    prefix: str = "cache",
) -> None:
    """Set a cache value with TTL."""
    r = get_redis()
    full_key = f"{prefix}:{key}"
    await r.set(full_key, value, ex=ttl_seconds)


async def cache_get(key: str, prefix: str = "cache") -> str | None:
    """Get a cached value. Returns None if not found or expired."""
    r = get_redis()
    full_key = f"{prefix}:{key}"
    return await r.get(full_key)


async def cache_delete(key: str, prefix: str = "cache") -> None:
    """Delete a cached value."""
    r = get_redis()
    full_key = f"{prefix}:{key}"
    await r.delete(full_key)


# ── Health Check ────────────────────────────

async def health_check() -> dict[str, Any]:
    """
    Check Redis health.
    
    Returns:
        Dictionary with health status, latency, and memory info.
    """
    import time

    try:
        r = get_redis()
        start = time.monotonic()
        await r.ping()
        latency_ms = (time.monotonic() - start) * 1000

        info = await r.info("memory")
        used_memory_mb = int(info.get("used_memory", 0)) / (1024 * 1024)

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "used_memory_mb": round(used_memory_mb, 2),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
