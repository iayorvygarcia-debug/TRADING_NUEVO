"""FRED (Federal Reserve Economic Data) Macroeconomic Regime Classifier."""

from datetime import datetime, timezone
from typing import Dict, Optional
import httpx
import structlog
from core.config import get_settings
from services.macro_sec.models import MacroIndicators, MacroRegime

logger = structlog.get_logger(__name__)


class FredMacroClassifier:
    """Classifies macroeconomic regime and calculates portfolio risk multipliers using FRED indicators."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.fred.api_key
        self.base_url = (base_url or settings.fred.base_url).rstrip("/")
        # In-memory cached indicators
        self._cached_indicators: Optional[MacroIndicators] = None

    def classify_regime(
        self,
        yield_curve_10y_2y: float,
        fed_funds_rate: float,
        vix_level: float,
        cpi_inflation: float = 2.8,
        unemployment_rate: Optional[float] = 4.1,
    ) -> MacroIndicators:
        """Determines the macro regime and risk multiplier from input metrics.

        Decision Rules:
        - VIX >= 28.0 OR (Yield Curve inverted < -0.30 and Fed Funds > 4.5) -> CONTRACTION_RISK_OFF (0.40x)
        - VIX >= 22.0 OR Yield Curve inverted (< 0.0) -> LATE_CYCLE_DEFENSIVE (0.70x)
        - VIX < 18.0 AND Yield Curve normal (>= 0.20) -> EXPANSION_RISK_ON (1.00x)
        - Otherwise -> RECOVERY_TRANSITION (0.85x)
        """
        if vix_level >= 28.0 or (yield_curve_10y_2y < -0.30 and fed_funds_rate > 4.5):
            regime = MacroRegime.CONTRACTION_RISK_OFF
            risk_multiplier = 0.40
        elif vix_level >= 22.0 or yield_curve_10y_2y < 0.0:
            regime = MacroRegime.LATE_CYCLE_DEFENSIVE
            risk_multiplier = 0.70
        elif vix_level < 18.0 and yield_curve_10y_2y >= 0.20:
            regime = MacroRegime.EXPANSION_RISK_ON
            risk_multiplier = 1.00
        else:
            regime = MacroRegime.RECOVERY_TRANSITION
            risk_multiplier = 0.85

        indicators = MacroIndicators(
            yield_curve_10y_2y=yield_curve_10y_2y,
            fed_funds_rate=fed_funds_rate,
            vix_level=vix_level,
            cpi_inflation_rate=cpi_inflation,
            unemployment_rate=unemployment_rate,
            regime=regime,
            risk_multiplier=risk_multiplier,
            as_of=datetime.now(timezone.utc),
        )
        self._cached_indicators = indicators
        return indicators

    async def fetch_series_latest_value(self, series_id: str) -> Optional[float]:
        """Fetches the latest observation for a given FRED series ID."""
        if not self.api_key:
            logger.info("fred_api_key_missing_using_default", series_id=series_id)
            return None

        url = f"{self.base_url}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    for obs in data.get("observations", []):
                        val_str = obs.get("value")
                        if val_str and val_str != ".":
                            return float(val_str)
        except Exception as exc:
            logger.warning("fred_fetch_failed", series_id=series_id, error=str(exc))

        return None

    async def update_macro_regime(self) -> MacroIndicators:
        """Fetches latest macro observations from FRED and updates current regime.

        If API is unreachable or unconfigured, falls back to safe moderate baseline values.
        """
        t10y2y = await self.fetch_series_latest_value("T10Y2Y")
        dff = await self.fetch_series_latest_value("DFF")
        vix = await self.fetch_series_latest_value("VIXCLS")

        # Fallback to sensible defaults if FRED is offline
        yield_spread = t10y2y if t10y2y is not None else 0.45
        fed_rate = dff if dff is not None else 4.25
        vix_val = vix if vix is not None else 16.5

        return self.classify_regime(
            yield_curve_10y_2y=yield_spread,
            fed_funds_rate=fed_rate,
            vix_level=vix_val,
        )

    def get_current_indicators(self) -> MacroIndicators:
        """Returns the cached macro indicators or a safe default."""
        if self._cached_indicators is None:
            return self.classify_regime(
                yield_curve_10y_2y=0.45,
                fed_funds_rate=4.25,
                vix_level=16.5,
            )
        return self._cached_indicators
