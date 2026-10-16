"""News & Sentiment Engine domain models and enumerations."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NewsCatalystType(str, Enum):
    """High-impact quantitative catalyst events (including 2026 AI & macro drivers)."""
    EARNINGS_BEAT = "EARNINGS_BEAT"
    EARNINGS_MISS = "EARNINGS_MISS"
    GUIDANCE_RAISE = "GUIDANCE_RAISE"
    GUIDANCE_LOWER = "GUIDANCE_LOWER"
    FDA_APPROVAL = "FDA_APPROVAL"
    FDA_REJECTION = "FDA_REJECTION"
    MERGER_ACQUISITION = "MERGER_ACQUISITION"
    ANALYST_UPGRADE = "ANALYST_UPGRADE"
    ANALYST_DOWNGRADE = "ANALYST_DOWNGRADE"
    DIVIDEND_INCREASE = "DIVIDEND_INCREASE"
    DIVIDEND_CUT = "DIVIDEND_CUT"
    LEGAL_REGULATORY_PROBE = "LEGAL_REGULATORY_PROBE"
    PATENT_INNOVATION = "PATENT_INNOVATION"
    AI_CHIP_CYCLE = "AI_CHIP_CYCLE"
    EXPORT_CONTROL_TARIFF = "EXPORT_CONTROL_TARIFF"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    SELL_THE_NEWS_DIVERGENCE = "SELL_THE_NEWS_DIVERGENCE"
    EARNINGS_REVISION = "EARNINGS_REVISION"
    GENERAL_FINANCIAL = "GENERAL_FINANCIAL"


class NewsCatalyst(BaseModel):
    """Extracted financial catalyst."""
    catalyst_type: NewsCatalystType
    direction: int = Field(..., description="+1 for bullish, -1 for bearish, 0 for neutral")
    weight: float = Field(default=0.8, ge=0.0, le=1.0, description="Impact weight of catalyst")
    headline_snippet: str = Field(..., description="Matched text snippet triggering catalyst")


class NewsArticle(BaseModel):
    """Normalized financial news article representation."""
    id: str
    headline: str
    summary: Optional[str] = None
    content: Optional[str] = None
    source: str = "Unknown"
    url: Optional[str] = None
    symbols: list[str] = Field(default_factory=list)
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # NLP & Quantitative Sentiment Attributes
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Sentiment polarity (-1.0 to +1.0)")
    sentiment_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence (0.0 to 1.0)")
    source_trust_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Reputational trust factor of source")
    catalysts: list[NewsCatalyst] = Field(default_factory=list, description="Extracted high-impact catalysts")
    is_sell_the_news: bool = Field(default=False, description="True if positive report met with adverse guidance/price reaction")
    recommended_action: str = Field(default="HOLD", description="Trading bias derived from Gemini/NLP engine")

    # Syndication Tracking
    syndication_count: int = Field(default=1, description="Number of duplicate/syndicated wire reports")


class SymbolSentimentState(BaseModel):
    """Aggregated sentiment and buzz state for a single ticker symbol."""
    symbol: str
    raw_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0, description="Average unweighted raw sentiment")
    decayed_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0, description="Exponentially time-decayed sentiment")
    article_count_24h: int = Field(default=0, description="News articles received in the last 24 hours")
    buzz_ratio: float = Field(default=1.0, description="Volume of news vs trailing 7-day average")
    bullish_catalysts_count: int = Field(default=0)
    bearish_catalysts_count: int = Field(default=0)
    active_catalysts: list[NewsCatalyst] = Field(default_factory=list)
    composite_news_score: float = Field(default=0.0, ge=-100.0, le=100.0, description="Composite score (-100 to +100)")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_bullish_catalyst(self) -> bool:
        return self.bullish_catalysts_count > self.bearish_catalysts_count and self.composite_news_score > 20.0

    @property
    def is_bearish_catalyst(self) -> bool:
        return self.bearish_catalysts_count > self.bullish_catalysts_count and self.composite_news_score < -20.0

    @property
    def is_buzz_spike(self) -> bool:
        return self.buzz_ratio >= 2.5 and self.article_count_24h >= 3
