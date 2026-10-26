"""Unit and integration tests for 24/7 Daemon Worker and FastAPI Telemetry/Control Endpoints."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import pytest
from apps.api.main import app
from apps.worker.quant_trading_worker import QuantTradingWorker, get_global_worker
from services.broker_engine.paper_broker import PaperBroker
from services.macro_sec.regime_manager import MarketRegimeManager
from services.risk_engine.circuit_breaker import get_circuit_breaker
from services.risk_engine.kill_switch import get_kill_switch


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def mock_db_and_redis():
    """Mocks async db and redis in FastAPI lifespan to prevent connection timeouts during test runs."""
    with patch("apps.api.main.init_database", new_callable=AsyncMock), \
         patch("apps.api.main.init_redis", new_callable=AsyncMock), \
         patch("apps.api.main.close_database", new_callable=AsyncMock), \
         patch("apps.api.main.close_redis", new_callable=AsyncMock):
        yield


@pytest.fixture(autouse=True)
def reset_singletons():
    """Resets Kill Switch and Circuit Breaker before and after each test."""
    ks = get_kill_switch()
    ks.reset()
    cb = get_circuit_breaker()
    cb.reset()
    yield
    ks.reset()
    cb.reset()


# ==========================================
# 1. 24/7 Autonomous Worker Tests
# ==========================================

@pytest.mark.anyio
async def test_quant_worker_telemetry_snapshot():
    worker = QuantTradingWorker()
    snap = worker.get_telemetry_snapshot()

    assert "portfolio" in snap
    assert snap["portfolio"]["cash"] == 100000.0
    assert snap["portfolio"]["portfolio_value"] == 100000.0
    assert snap["kill_switch_active"] is False
    assert "macro" in snap
    assert "circuit_breaker_tier" in snap


@pytest.mark.anyio
async def test_quant_worker_cycle_skips_when_kill_switch_active():
    worker = QuantTradingWorker()
    ks = get_kill_switch()
    ks.trigger(reason="Test Halt")

    snap = await worker.execute_cycle()
    assert snap["kill_switch_active"] is True
    assert snap["kill_switch_reason"] == "Test Halt"


@pytest.mark.anyio
async def test_quant_worker_sec_adverse_block():
    worker = QuantTradingWorker()
    worker.regime_manager._blocked_symbols.add("BADCO")

    assert worker.regime_manager.is_symbol_blocked("BADCO") is True


# ==========================================
# 2. FastAPI Gateway & Telemetry Endpoints
# ==========================================

def test_api_health_endpoint():
    with TestClient(app) as client:
        with patch("core.database.health_check", new_callable=AsyncMock, return_value={"status": "healthy"}), \
             patch("core.redis_client.health_check", new_callable=AsyncMock, return_value={"status": "healthy"}), \
             patch("core.event_bus.event_bus_health", new_callable=AsyncMock, return_value={"status": "healthy"}):
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "trading_mode" in data


def test_api_status_endpoint():
    with TestClient(app) as client:
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == "AI Quant Trading Platform"
        assert "risk_config" in data


def test_api_serve_dashboard_html():
    with TestClient(app) as client:
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "ANTIGRAVITY // QUANT OS" in resp.text
        assert "KILL SWITCH" in resp.text


def test_api_telemetry_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/v1/telemetry")
        assert resp.status_code == 200
        data = resp.json()
        assert "portfolio" in data
        assert "macro" in data
        assert "kill_switch_active" in data
        assert data["kill_switch_active"] is False


def test_api_kill_switch_activation_and_resume():
    with TestClient(app) as client:
        # 1. Trigger Kill Switch
        resp_kill = client.post(
            "/api/v1/control/kill-switch",
            json={"reason": "Volatility surge spike"},
        )
        assert resp_kill.status_code == 200
        data_kill = resp_kill.json()
        assert data_kill["status"] == "kill_switch_activated"
        assert data_kill["is_active"] is True

        # Telemetry should reflect kill switch active
        resp_telemetry = client.get("/api/v1/telemetry")
        assert resp_telemetry.json()["kill_switch_active"] is True

        # 2. Attempt resume with invalid passphrase -> 403 Forbidden
        resp_bad_resume = client.post(
            "/api/v1/control/resume",
            json={"passphrase": "WRONG_PASSWORD"},
        )
        assert resp_bad_resume.status_code == 403

        # 3. Resume with correct passphrase CONFIRM_RESUME -> 200 OK
        resp_good_resume = client.post(
            "/api/v1/control/resume",
            json={"passphrase": "CONFIRM_RESUME"},
        )
        assert resp_good_resume.status_code == 200
        assert resp_good_resume.json()["status"] == "trading_resumed"
        assert resp_good_resume.json()["kill_switch_active"] is False


def test_api_on_demand_scan_trigger():
    with TestClient(app) as client:
        # Mock scanner to run quickly
        with patch.object(get_global_worker().scanner, "scan_universe", new_callable=AsyncMock, return_value=[]):
            resp = client.post("/api/v1/control/scan")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "cycle_completed"
            assert "telemetry" in data


def test_api_toggle_shadow_mode():
    with TestClient(app) as client:
        resp = client.post("/api/v1/control/shadow-mode", json={"enabled": True})
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"
        assert resp.json()["shadow_mode"] is True

        resp_off = client.post("/api/v1/control/shadow-mode", json={"enabled": False})
        assert resp_off.status_code == 200
        assert resp_off.json()["shadow_mode"] is False


def test_api_circuit_breaker_reset():
    with TestClient(app) as client:
        resp = client.post("/api/v1/control/circuit-breaker/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "circuit_breaker_reset"
        assert data["circuit_breaker_tier"] == "NORMAL"

