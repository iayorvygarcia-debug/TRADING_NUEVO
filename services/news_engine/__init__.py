"""News & Sentiment Engine package."""

from services.news_engine.models import (
    NewsArticle,
    NewsCatalyst,
    NewsCatalystType,
    SymbolSentimentState,
)
from services.news_engine.sentiment import (
    BaseSentimentAnalyzer,
    FinancialLexiconAnalyzer,
)
from services.news_engine.catalyst_extractor import CatalystExtractor
from services.news_engine.dedup import NewsDeduplicator, SourceTrustRater
from services.news_engine.impact import NewsImpactEngine
from services.news_engine.alpaca_news import AlpacaNewsAdapter

__all__ = [
    "NewsArticle",
    "NewsCatalyst",
    "NewsCatalystType",
    "SymbolSentimentState",
    "BaseSentimentAnalyzer",
    "FinancialLexiconAnalyzer",
    "CatalystExtractor",
    "NewsDeduplicator",
    "SourceTrustRater",
    "NewsImpactEngine",
    "AlpacaNewsAdapter",
]
