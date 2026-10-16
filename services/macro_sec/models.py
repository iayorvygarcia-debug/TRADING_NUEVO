"""Domain models for Macro Regime & SEC EDGAR Intelligence."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class MacroRegime(str, Enum):
    """Global macroeconomic regime classification."""
    EXPANSION_RISK_ON = "EXPANSION_RISK_ON"
    LATE_CYCLE_DEFENSIVE = "LATE_CYCLE_DEFENSIVE"
    CONTRACTION_RISK_OFF = "CONTRACTION_RISK_OFF"
    RECOVERY_TRANSITION = "RECOVERY_TRANSITION"


class MacroIndicators(BaseModel):
    """Macroeconomic data points and derived regime."""
    yield_curve_10y_2y: float = Field(..., description="10Y minus 2Y Treasury yield spread in %")
    fed_funds_rate: float = Field(..., description="Effective Federal Funds rate in %")
    vix_level: float = Field(..., description="CBOE Volatility Index level")
    cpi_inflation_rate: float = Field(..., description="Trailing 12-month CPI inflation %")
    unemployment_rate: Optional[float] = Field(None, description="US headline unemployment rate %")
    regime: MacroRegime = Field(default=MacroRegime.EXPANSION_RISK_ON)
    risk_multiplier: float = Field(default=1.0, ge=0.1, le=1.0, description="Position size scaling factor")
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_yield_curve_inverted(self) -> bool:
        return self.yield_curve_10y_2y < 0.0

    @property
    def is_high_volatility(self) -> bool:
        return self.vix_level >= 28.0

    @property
    def is_low_volatility(self) -> bool:
        return self.vix_level < 16.0


class FilingType(str, Enum):
    """SEC Form types relevant to quantitative trading."""
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    FORM_4 = "4"
    OTHER = "OTHER"


class InsiderTransaction(BaseModel):
    """Executive & Director insider transaction record (SEC Form 4)."""
    symbol: str
    insider_name: str
    insider_title: str
    transaction_code: str = Field(..., description="'P'=Open Market Purchase, 'S'=Open Market Sale, 'M'=Option Exercise, 'A'=Award")
    shares: float
    price_per_share: float
    total_value: float
    transaction_date: datetime
    filing_date: datetime

    @property
    def is_cluster_worthy_purchase(self) -> bool:
        """True if open-market purchase by C-suite/Director over $50k."""
        is_exec = any(title in self.insider_title.upper() for title in ["CEO", "CFO", "CHIEF", "DIRECTOR", "PRESIDENT"])
        return self.transaction_code == "P" and self.total_value >= 50000.0 and is_exec


class SECCompanyFiling(BaseModel):
    """SEC EDGAR corporate filing record."""
    cik: str
    symbol: str
    filing_type: FilingType
    filing_date: datetime
    accession_number: str
    primary_doc_url: str
    items_reported: List[str] = Field(default_factory=list, description="Reported items, especially for 8-Ks")
    is_material_adverse_event: bool = Field(default=False, description="Whether 8-K contains distress/departure/default")
