"""
Tests for Twelve Data Adapter and Key Rotator.

Validates:
- Multi-key rotation (round-robin)
- Per-key rate limiting & credit tracking
- Key exhaustion handling
- Indicator & time series parsing
- 429 retry logic with key switching
- Health checks
"""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from services.market_data.twelve_data_adapter import (
    KeyRotator,
    TwelveDataAdapter,
    AVAILABLE_INDICATORS,
)


class TestKeyRotator:
    """Test multi-key rotation and limit tracking."""

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="At least one API key is required"):
            KeyRotator(keys=[])

    def test_initial_state(self) -> None:
        rotator = KeyRotator(
            keys=["key_a", "key_b", "key_c"],
            rate_per_key=8,
            credits_per_key=800,
        )
        assert rotator.num_keys == 3
        assert rotator.total_credits_remaining == 2400

        stats = rotator.get_stats()
        assert stats["num_keys"] == 3
        assert stats["total_credits_remaining"] == 2400
        assert "key_1" in stats["credits_per_key"]

    @pytest.mark.anyio
    async def test_round_robin_distribution(self) -> None:
        rotator = KeyRotator(
            keys=["k1", "k2", "k3"],
            rate_per_key=8,
            credits_per_key=10,
        )
        # Should cycle k1 -> k2 -> k3 -> k1
        assert await rotator.get_key() == "k1"
        assert await rotator.get_key() == "k2"
        assert await rotator.get_key() == "k3"
        assert await rotator.get_key() == "k1"

    @pytest.mark.anyio
    async def test_credit_tracking(self) -> None:
        rotator = KeyRotator(
            keys=["k1", "k2"],
            rate_per_key=8,
            credits_per_key=5,
        )
        assert rotator.total_credits_remaining == 10

        await rotator.get_key()  # k1 (used 1)
        await rotator.get_key()  # k2 (used 1)

        assert rotator.total_credits_remaining == 8
        stats = rotator.get_stats()
        assert stats["credits_per_key"]["key_1"]["used"] == 1
        assert stats["credits_per_key"]["key_1"]["remaining"] == 4

    @pytest.mark.anyio
    async def test_exhausted_key_skipped(self) -> None:
        rotator = KeyRotator(
            keys=["k1", "k2"],
            rate_per_key=8,
            credits_per_key=2,
        )
        # Exhaust k1
        assert await rotator.get_key() == "k1"  # k1: 1
        assert await rotator.get_key() == "k2"  # k2: 1
        assert await rotator.get_key() == "k1"  # k1: 2 (now exhausted!)
        
        # Next call should pick k2 because k1 is exhausted
        assert await rotator.get_key() == "k2"  # k2: 2 (now exhausted!)

        # Now both are exhausted
        with pytest.raises(RuntimeError, match="exhausted for today"):
            await rotator.get_key()

    def test_daily_reset(self) -> None:
        rotator = KeyRotator(keys=["k1"], credits_per_key=10)
        rotator._daily_credits[0] = 10
        assert rotator.total_credits_remaining == 0

        # Simulate date change
        rotator._credits_date = date(2026, 1, 1)
        rotator._maybe_reset_daily()
        assert rotator.total_credits_remaining == 10


class TestTwelveDataAdapter:
    """Test Twelve Data HTTP adapter methods with mocks."""

    @pytest.fixture
    def anyio_backend(self) -> str:
        return "asyncio"

    @pytest.fixture
    def adapter(self) -> TwelveDataAdapter:
        return TwelveDataAdapter()

    @pytest.mark.anyio
    async def test_connect_without_keys_raises(self, adapter: TwelveDataAdapter) -> None:
        with patch.object(type(adapter._settings), "get_active_keys", return_value=[]):
            with pytest.raises(RuntimeError, match="No Twelve Data API keys configured"):
                await adapter.connect()

    @pytest.mark.anyio
    async def test_connect_and_disconnect(self, adapter: TwelveDataAdapter) -> None:
        with patch.object(type(adapter._settings), "get_active_keys", return_value=["key1", "key2"]):
            await adapter.connect()
            assert adapter._client is not None
            assert adapter._rotator is not None
            assert adapter._rotator.num_keys == 2

            await adapter.disconnect()
            assert adapter._client is None

    @pytest.mark.anyio
    async def test_get_time_series(self, adapter: TwelveDataAdapter) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"symbol": "AAPL", "interval": "1day"},
            "values": [
                {
                    "datetime": "2026-09-08",
                    "open": "220.00",
                    "high": "225.00",
                    "low": "219.00",
                    "close": "224.50",
                    "volume": "50000000",
                }
            ],
            "status": "ok",
        }

        with patch.object(type(adapter._settings), "get_active_keys", return_value=["dummy_key"]):
            await adapter.connect()
            with patch.object(adapter._client, "get", new=AsyncMock(return_value=mock_response)):
                bars = await adapter.get_time_series("AAPL", interval="1day", outputsize=1)
                assert len(bars) == 1
                assert bars[0]["close"] == "224.50"
            await adapter.disconnect()

    @pytest.mark.anyio
    async def test_get_indicator(self, adapter: TwelveDataAdapter) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"symbol": "AAPL", "indicator": {"name": "RSI"}},
            "values": [
                {"datetime": "2026-09-08", "rsi": "58.23"},
                {"datetime": "2026-09-07", "rsi": "55.10"},
            ],
            "status": "ok",
        }

        with patch.object(type(adapter._settings), "get_active_keys", return_value=["dummy_key"]):
            await adapter.connect()
            with patch.object(adapter._client, "get", new=AsyncMock(return_value=mock_response)):
                values = await adapter.get_indicator("AAPL", "rsi", time_period=14)
                assert len(values) == 2
                assert values[0]["rsi"] == "58.23"
            await adapter.disconnect()

    @pytest.mark.anyio
    async def test_get_quote(self, adapter: TwelveDataAdapter) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "close": "224.50",
            "change": "2.50",
            "percent_change": "1.13",
        }

        with patch.object(type(adapter._settings), "get_active_keys", return_value=["dummy_key"]):
            await adapter.connect()
            with patch.object(adapter._client, "get", new=AsyncMock(return_value=mock_response)):
                quote = await adapter.get_quote("AAPL")
                assert quote["symbol"] == "AAPL"
                assert quote["close"] == "224.50"
            await adapter.disconnect()

    @pytest.mark.anyio
    async def test_rate_limit_retry_with_next_key(self, adapter: TwelveDataAdapter) -> None:
        resp_429 = MagicMock()
        resp_429.status_code = 200
        resp_429.json.return_value = {
            "status": "error",
            "code": 429,
            "message": "You have run out of API credits for this key.",
        }

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {
            "symbol": "MSFT",
            "close": "450.00",
        }

        with patch.object(type(adapter._settings), "get_active_keys", return_value=["key1", "key2"]):
            await adapter.connect()
            with patch.object(adapter._client, "get", new=AsyncMock(side_effect=[resp_429, resp_ok])):
                quote = await adapter.get_quote("MSFT")
                assert quote["symbol"] == "MSFT"
                assert quote["close"] == "450.00"
            await adapter.disconnect()
