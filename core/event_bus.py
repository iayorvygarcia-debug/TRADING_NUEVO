"""
AI Quant Trading Platform — Event Bus (Redis Streams)

Event-driven architecture using Redis Streams.

Why Redis Streams (not Kafka):
- Already using Redis for cache → no extra infrastructure
- Persistent, ordered log with consumer groups
- Sufficient throughput for MVP (~1000 events/sec)
- Simpler operations than Kafka cluster
- Migration path to Kafka is well-documented if needed

How it works:
1. Producers publish typed events to named streams
2. Consumer groups read events with guaranteed delivery
3. Each consumer in a group gets unique events (parallel processing)
4. Failed events can be claimed and retried (pending entries list)

Architecture:
    Producer → Redis Stream → Consumer Group → Consumer
    
    MarketDataWorker → "stream:market" → "signal-group" → SignalWorker
    NewsWorker       → "stream:news"   → "signal-group" → SignalWorker
                                       → "audit-group"  → AuditWorker
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

from core.redis_client import get_redis
from core.logging import get_logger

logger = get_logger("event_bus")


# ── Event Types ─────────────────────────────

class EventType(str, Enum):
    """All event types in the system. Typed for safety."""

    # Market Data
    MARKET_TICK = "market.tick"
    BAR_CLOSED = "market.bar_closed"
    QUOTE_UPDATED = "market.quote_updated"

    # News
    NEWS_RECEIVED = "news.received"
    NEWS_EVENT_DETECTED = "news.event_detected"

    # SEC / Macro
    SEC_FILING_RECEIVED = "sec.filing_received"
    MACRO_UPDATED = "macro.updated"

    # Intelligence
    FEATURE_COMPUTED = "intelligence.feature_computed"
    REGIME_CHANGED = "intelligence.regime_changed"

    # Signals
    SIGNAL_GENERATED = "signal.generated"
    OPPORTUNITY_RANKED = "signal.opportunity_ranked"

    # Risk
    RISK_APPROVED = "risk.approved"
    RISK_REJECTED = "risk.rejected"
    CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker"
    KILL_SWITCH_ACTIVATED = "risk.kill_switch"

    # Orders
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_PARTIALLY_FILLED = "order.partially_filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"

    # Positions
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"

    # System
    SYSTEM_HEALTH_CHANGED = "system.health_changed"
    MODEL_DRIFT_DETECTED = "system.model_drift"


@dataclass
class Event:
    """
    A typed event in the system.
    
    Every event has:
    - type: What happened (from EventType enum)
    - data: Event payload (JSON-serializable dict)
    - event_id: Unique identifier
    - timestamp: When the event was created
    - source: Which service produced it
    """
    type: EventType
    data: dict[str, Any]
    source: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, str]:
        """Serialize for Redis Stream (all values must be strings)."""
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "source": self.source,
            "timestamp": str(self.timestamp),
            "data": json.dumps(self.data),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> Event:
        """Deserialize from Redis Stream."""
        return cls(
            event_id=raw["event_id"],
            type=EventType(raw["type"]),
            source=raw["source"],
            timestamp=float(raw["timestamp"]),
            data=json.loads(raw["data"]),
        )


# ── Stream Names ────────────────────────────

def _stream_name(event_type: EventType) -> str:
    """Map event type to Redis stream name."""
    # Group by domain: market.*, news.*, signal.*, etc.
    domain = event_type.value.split(".")[0]
    return f"stream:{domain}"


# ── Publisher ───────────────────────────────

async def publish(event: Event) -> str:
    """
    Publish an event to the appropriate Redis Stream.
    
    Args:
        event: The event to publish.
    
    Returns:
        The Redis message ID.
    """
    try:
        from core.redis_client import is_redis_available
        if not is_redis_available():
            return ""
        redis = get_redis()
    except RuntimeError:
        return ""

    stream = _stream_name(event.type)

    try:
        import asyncio
        message_id = await asyncio.wait_for(
            redis.xadd(
                stream,
                event.to_dict(),
                maxlen=100_000,  # Keep last 100K events per stream (auto-trim)
            ),
            timeout=0.5,
        )

        logger.debug(
            "event_published",
            event_type=event.type.value,
            stream=stream,
            message_id=message_id,
            source=event.source,
        )

        return message_id
    except Exception as exc:
        logger.debug("event_publish_skipped", stream=stream, error=str(exc))
        return ""


async def publish_event(
    event_type: EventType,
    data: dict[str, Any],
    source: str,
) -> str:
    """
    Convenience function to create and publish an event.
    
    Args:
        event_type: Type of the event.
        data: Event payload.
        source: Service name that produced the event.
    
    Returns:
        The Redis message ID.
    """
    event = Event(type=event_type, data=data, source=source)
    return await publish(event)


# ── Consumer ────────────────────────────────

EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


async def ensure_consumer_group(
    stream: str,
    group: str,
    start_id: str = "0",
) -> None:
    """Create a consumer group if it doesn't exist."""
    redis = get_redis()
    try:
        await redis.xgroup_create(stream, group, id=start_id, mkstream=True)
        logger.info("consumer_group_created", stream=stream, group=group)
    except Exception as e:
        if "BUSYGROUP" in str(e):
            pass  # Group already exists, that's fine
        else:
            raise


async def consume_events(
    stream: str,
    group: str,
    consumer: str,
    handler: EventHandler,
    batch_size: int = 10,
    block_ms: int = 5000,
) -> int:
    """
    Read and process events from a stream using consumer groups.
    
    This is called in a loop by worker services.
    
    Args:
        stream: Redis stream name (e.g., "stream:market").
        group: Consumer group name.
        consumer: Consumer name within the group.
        handler: Async function to process each event.
        batch_size: Max events to read per call.
        block_ms: How long to block waiting for new events.
    
    Returns:
        Number of events processed.
    """
    redis = get_redis()
    processed = 0

    # Read new messages for this consumer
    messages = await redis.xreadgroup(
        groupname=group,
        consumername=consumer,
        streams={stream: ">"},  # ">" = only new, undelivered messages
        count=batch_size,
        block=block_ms,
    )

    if not messages:
        return 0

    for _stream_name_bytes, entries in messages:
        for message_id, raw_data in entries:
            try:
                event = Event.from_dict(raw_data)
                await handler(event)

                # Acknowledge successful processing
                await redis.xack(stream, group, message_id)
                processed += 1

            except Exception as e:
                logger.error(
                    "event_processing_failed",
                    stream=stream,
                    message_id=message_id,
                    error=str(e),
                )
                # Don't ACK — message stays in pending entries list (PEL)
                # Can be claimed and retried later

    return processed


# ── Health Check ────────────────────────────

async def event_bus_health() -> dict[str, Any]:
    """Check event bus health — stream lengths and consumer group lag."""
    try:
        from core.redis_client import is_redis_available
        if not is_redis_available():
            return {"status": "degraded", "error": "Redis offline (in-memory mode)", "streams": {}}
        redis = get_redis()
    except Exception as exc:
        return {"status": "degraded", "error": str(exc), "streams": {}}

    health: dict[str, Any] = {"status": "healthy", "streams": {}}

    try:
        # Check all known streams
        for stream_name in ["stream:market", "stream:news", "stream:signal",
                            "stream:order", "stream:position", "stream:risk",
                            "stream:system", "stream:intelligence",
                            "stream:sec", "stream:macro"]:
            try:
                length = await redis.xlen(stream_name)
                health["streams"][stream_name] = {"length": length}
            except Exception:
                health["streams"][stream_name] = {"length": 0, "exists": False}

    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)

    return health
