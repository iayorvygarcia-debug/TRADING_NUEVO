"""
Tests for core configuration.

Validates:
- Default trading mode is PAPER
- Settings load correctly from defaults
- Risk defaults are safe
"""

from __future__ import annotations

import pytest

from core.config import (
    Settings,
    AlpacaSettings,
    TradingMode,
    Environment,
    reset_settings,
    get_settings,
)


class TestTradingMode:
    """Trading mode must default to PAPER."""

    def setup_method(self):
        reset_settings()

    def test_default_is_paper(self):
        """Default trading mode MUST be paper."""
        settings = Settings()
        assert settings.trading_mode == TradingMode.PAPER

    def test_is_paper_property(self):
        settings = Settings()
        assert settings.is_paper is True
        assert settings.is_live is False

    def test_live_warning(self):
        """Setting LIVE should produce a warning."""
        with pytest.warns(UserWarning, match="LIVE TRADING MODE"):
            Settings(trading_mode="live")


class TestRiskDefaults:
    """Risk defaults must be conservative."""

    def test_risk_per_trade(self):
        settings = Settings()
        assert settings.risk.risk_per_trade == 0.01  # 1% max

    def test_max_daily_loss(self):
        settings = Settings()
        assert settings.risk.max_daily_loss == 0.03  # 3% max

    def test_max_drawdown(self):
        settings = Settings()
        assert settings.risk.max_drawdown == 0.10  # 10% max

    def test_max_position_size(self):
        settings = Settings()
        assert settings.risk.max_position_size == 0.05  # 5% max


class TestAlpacaConfig:
    """Alpaca settings validation."""

    def test_default_is_paper_url(self):
        settings = Settings()
        assert "paper" in settings.alpaca.base_url
        assert settings.alpaca.is_paper is True

    def test_keys_not_set_by_default(self):
        alpaca = AlpacaSettings(api_key="", secret_key="")
        assert alpaca.validate_keys() is False
        alpaca_valid = AlpacaSettings(api_key="test_k", secret_key="test_s")
        assert alpaca_valid.validate_keys() is True


class TestEnvironment:
    """Environment defaults."""

    def test_default_development(self):
        settings = Settings()
        assert settings.environment == Environment.DEVELOPMENT

    def test_default_log_level(self):
        settings = Settings()
        assert settings.log_level == "INFO"
