"""
AI Quant Trading Platform — Core Configuration

Centralized configuration management using pydantic-settings (with fallback
to standard pydantic BaseSettings when running in minimal environments).
Includes GeminiSettings for Google Gemini AI News & Market Intelligence.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    from pydantic import field_validator
except ImportError:
    from pydantic import validator as _validator

    def field_validator(*fields: str, mode: str = "before"):  # type: ignore
        return _validator(*fields, pre=(mode == "before"), allow_reuse=True)

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:
    try:
        from pydantic import BaseSettings  # type: ignore
    except ImportError:
        BaseSettings = BaseModel  # type: ignore
    SettingsConfigDict = dict  # type: ignore
    _HAS_PYDANTIC_SETTINGS = False


class TradingMode(str, Enum):
    """Trading mode enum. PAPER is mandatory during development."""
    PAPER = "paper"
    LIVE = "live"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """PostgreSQL + TimescaleDB configuration."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="DATABASE_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    host: str = "localhost"
    port: int = 5432
    name: str = "trading_platform"
    user: str = "trading"
    password: str = "changeme"

    @property
    def async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def sync_url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class RedisSettings(BaseSettings):
    """Redis configuration for cache and event bus."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="REDIS_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class AlpacaSettings(BaseSettings):
    """Alpaca Markets broker/data configuration."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="ALPACA_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    api_key: str = Field(default_factory=lambda: os.environ.get("ALPACA_API_KEY", ""))
    secret_key: str = Field(default_factory=lambda: os.environ.get("ALPACA_SECRET_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
    data_url: str = "https://data.alpaca.markets"
    stream_url: str = "wss://stream.data.alpaca.markets/v2"
    feed: str = "iex"

    @property
    def is_paper(self) -> bool:
        return "paper" in self.base_url

    def validate_keys(self) -> bool:
        return bool(self.api_key and self.secret_key)


class GeminiSettings(BaseSettings):
    """Google Gemini AI configuration for News & Catalyst Intelligence."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="GEMINI_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    api_key: str = Field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    )
    model_name: str = Field(
        default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    )
    enabled: bool = True
    timeout_seconds: float = 6.0

    def validate_key(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)


class SECSettings(BaseSettings):
    """SEC EDGAR API configuration. No API key required."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="SEC_EDGAR_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    user_agent: str = "AIQuantPlatform contact@example.com"
    rate_limit: int = 10
    base_url: str = "https://data.sec.gov"
    filings_url: str = "https://efts.sec.gov/LATEST"


class FREDSettings(BaseSettings):
    """FRED API configuration. Free API key required."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="FRED_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    api_key: str = ""
    base_url: str = "https://api.stlouisfed.org/fred"
    rate_limit: int = 120


class TwelveDataSettings(BaseSettings):
    """Twelve Data API configuration with multi-key rotation."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="TWELVE_DATA_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    api_key: str = ""
    api_key_2: str = ""
    api_key_3: str = ""
    api_key_4: str = ""
    api_key_5: str = ""
    base_url: str = "https://api.twelvedata.com"
    credits_per_key: int = 800
    rate_limit_per_key: int = 8

    def get_active_keys(self) -> list[str]:
        all_keys = [
            self.api_key, self.api_key_2, self.api_key_3,
            self.api_key_4, self.api_key_5,
        ]
        return [k for k in all_keys if k]

    @property
    def total_daily_credits(self) -> int:
        return len(self.get_active_keys()) * self.credits_per_key

    @property
    def aggregate_rate_limit(self) -> int:
        return len(self.get_active_keys()) * self.rate_limit_per_key


class NotificationSettings(BaseSettings):
    """Notification channels configuration."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_to: str = ""

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def email_enabled(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.alert_email_to)


class RiskSettings(BaseSettings):
    """Risk engine configuration."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_prefix="",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01          # 1% max risk per trade
    max_daily_loss: float = 0.03          # 3% max daily loss
    max_weekly_loss: float = 0.05         # 5% max weekly loss
    max_drawdown: float = 0.10            # 10% max portfolio drawdown
    max_position_size: float = 0.05       # 5% max single position
    max_sector_exposure: float = 0.25     # 25% max sector exposure
    max_trades_per_day: int = 20
    max_orders_per_minute: int = 5
    max_spread_pct: float = 0.005         # 0.5% max spread
    min_signal_score: float = 0.65        # Minimum composite score
    min_probability: float = 0.55         # Minimum ML probability
    operational_capital: float = 1_000.0
    max_operational_position_size: float = 0.20
    min_profit_lock_usd: float = 20.0
    max_profit_giveback_pct: float = 35.0

    # Dynamic Spread Configuration
    spread_regular_base_pct: float = 0.0020
    spread_extended_multiplier: float = 1.5
    spread_max_hard_cap_pct: float = 0.0050
    spread_extended_hard_cap_pct: float = 0.0070

    @property
    def risk_per_trade_pct(self) -> float:
        """Alias to prevent attribute mismatch in ExecutionEngine."""
        return self.risk_per_trade


class Settings(BaseSettings):
    """Master configuration for the AI Quant Trading Platform."""
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    debug: bool = False
    trading_mode: TradingMode = Field(
        default_factory=lambda: TradingMode(os.environ.get("TRADING_MODE", "paper").lower())
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    twelve_data: TwelveDataSettings = Field(default_factory=TwelveDataSettings)
    sec: SECSettings = Field(default_factory=SECSettings)
    fred: FREDSettings = Field(default_factory=FREDSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)

    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @field_validator("trading_mode", mode="before")
    @classmethod
    def validate_trading_mode(cls, v: Any) -> str:
        if isinstance(v, str) and v.lower() == "live":
            import warnings
            warnings.warn(
                "⚠️  LIVE TRADING MODE requested. "
                "Ensure ALL safety checklist items are validated.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @property
    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == TradingMode.PAPER


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
