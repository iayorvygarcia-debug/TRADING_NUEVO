"""Unit tests for News & Sentiment Engine."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from services.news_engine.alpaca_news import AlpacaNewsAdapter
from services.news_engine.catalyst_extractor import CatalystExtractor
from services.news_engine.dedup import NewsDeduplicator, SourceTrustRater
from services.news_engine.impact import NewsImpactEngine
from services.news_engine.models import (
    NewsArticle,
    NewsCatalystType,
)
from services.news_engine.sentiment import FinancialLexiconAnalyzer


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ==========================================
# 1. Financial Lexicon Sentiment Tests
# ==========================================

def test_financial_sentiment_positive_earnings():
    analyzer = FinancialLexiconAnalyzer()
    score, conf = analyzer.analyze("Apple beats Q3 earnings estimates with record revenue growth")
    assert score > 0.5
    assert conf > 0.5


def test_financial_sentiment_crushed_expectations():
    analyzer = FinancialLexiconAnalyzer()
    score, conf = analyzer.analyze("NVIDIA crushed earnings expectations as data center demand surged")
    assert score >= 0.7
    assert conf > 0.5


def test_financial_sentiment_negative_earnings_miss():
    analyzer = FinancialLexiconAnalyzer()
    score, conf = analyzer.analyze("Tesla missed revenue estimates and slashes full-year vehicle delivery guidance")
    assert score < -0.5
    assert conf > 0.5


def test_financial_sentiment_negation():
    analyzer = FinancialLexiconAnalyzer()
    # "did not miss" should be inverted/neutralized compared to "missed"
    score_negated, _ = analyzer.analyze("The company did not miss quarterly targets")
    score_direct, _ = analyzer.analyze("The company missed quarterly targets")
    assert score_negated > score_direct


def test_financial_sentiment_intensifiers():
    analyzer = FinancialLexiconAnalyzer()
    score_normal, _ = analyzer.analyze("Quarterly profits surged")
    score_intensified, _ = analyzer.analyze("Quarterly profits substantially surged")
    assert abs(score_intensified) >= abs(score_normal)


def test_financial_sentiment_empty_or_neutral():
    analyzer = FinancialLexiconAnalyzer()
    score, conf = analyzer.analyze("")
    assert score == 0.0
    assert conf == 0.0

    score_neutral, conf_neutral = analyzer.analyze("The annual meeting will be held on Tuesday morning")
    assert score_neutral == 0.0


# ==========================================
# 2. Catalyst Extractor Tests
# ==========================================

def test_catalyst_extractor_earnings_beat():
    extractor = CatalystExtractor()
    cats = extractor.extract("Microsoft beats on earnings and revenue for fiscal Q4")
    assert len(cats) >= 1
    assert cats[0].catalyst_type == NewsCatalystType.EARNINGS_BEAT
    assert cats[0].direction == 1


def test_catalyst_extractor_guidance_raise():
    extractor = CatalystExtractor()
    cats = extractor.extract("AMD raises full-year guidance citing AI chip demand")
    assert any(c.catalyst_type == NewsCatalystType.GUIDANCE_RAISE for c in cats)
    assert any(c.direction == 1 for c in cats)


def test_catalyst_extractor_guidance_lower():
    extractor = CatalystExtractor()
    cats = extractor.extract("Intel cuts full-year forecast and warns on data center weakness")
    assert any(c.catalyst_type == NewsCatalystType.GUIDANCE_LOWER for c in cats)
    assert any(c.direction == -1 for c in cats)


def test_catalyst_extractor_fda_approval():
    extractor = CatalystExtractor()
    cats = extractor.extract("Biogen receives FDA approval for new therapy")
    assert any(c.catalyst_type == NewsCatalystType.FDA_APPROVAL for c in cats)
    assert any(c.direction == 1 for c in cats)


def test_catalyst_extractor_analyst_upgrade():
    extractor = CatalystExtractor()
    cats = extractor.extract("Goldman Sachs upgraded to Buy with price target raised to $350")
    assert any(c.catalyst_type == NewsCatalystType.ANALYST_UPGRADE for c in cats)


def test_catalyst_extractor_legal_probe():
    extractor = CatalystExtractor()
    cats = extractor.extract("DOJ launches antitrust investigation into Alphabet search agreements")
    assert any(c.catalyst_type == NewsCatalystType.LEGAL_REGULATORY_PROBE for c in cats)
    assert any(c.direction == -1 for c in cats)


def test_catalyst_extractor_dividend_actions():
    extractor = CatalystExtractor()
    cats_raise = extractor.extract("ExxonMobil raises quarterly dividend by 8%")
    assert any(c.catalyst_type == NewsCatalystType.DIVIDEND_INCREASE for c in cats_raise)

    cats_cut = extractor.extract("Walgreens slashes quarterly dividend to preserve liquidity")
    assert any(c.catalyst_type == NewsCatalystType.DIVIDEND_CUT for c in cats_cut)


# ==========================================
# 3. Deduplication & Trust Scoring Tests
# ==========================================

def test_source_trust_rater():
    assert SourceTrustRater.get_trust_score("Bloomberg News") == 1.0
    assert SourceTrustRater.get_trust_score("Reuters") == 1.0
    assert SourceTrustRater.get_trust_score("CNBC") == 0.85
    assert SourceTrustRater.get_trust_score("Seeking Alpha") == 0.65
    assert SourceTrustRater.get_trust_score("Random Trading Blog") == 0.50


def test_deduplicator_filters_syndicated_reprints():
    dedup = NewsDeduplicator(similarity_threshold=0.60)
    now = datetime.now(timezone.utc)

    art1 = NewsArticle(
        id="1",
        headline="Apple reports record quarterly revenue beating Wall Street estimates",
        source="Reuters",
        symbols=["AAPL"],
        published_at=now,
    )
    # Similar syndicated reprint
    art2 = NewsArticle(
        id="2",
        headline="Apple reports record quarterly revenue and beats Wall Street consensus",
        source="Benzinga",
        symbols=["AAPL"],
        published_at=now + timedelta(minutes=15),
    )

    result1 = dedup.process_and_deduplicate(art1)
    assert result1 is not None
    assert result1.id == "1"

    result2 = dedup.process_and_deduplicate(art2)
    # art2 is a duplicate, so result2 should be None and art1's syndication count incremented
    assert result2 is None
    assert art1.syndication_count == 2


def test_deduplicator_allows_distinct_news():
    dedup = NewsDeduplicator()
    now = datetime.now(timezone.utc)

    art1 = NewsArticle(
        id="1",
        headline="Tesla announces massive factory expansion in Austin",
        source="Bloomberg",
        symbols=["TSLA"],
        published_at=now,
    )
    art2 = NewsArticle(
        id="2",
        headline="Tesla faces regulatory probe over automated driving software",
        source="WSJ",
        symbols=["TSLA"],
        published_at=now + timedelta(minutes=30),
    )

    r1 = dedup.process_and_deduplicate(art1)
    r2 = dedup.process_and_deduplicate(art2)
    assert r1 is not None
    assert r2 is not None


# ==========================================
# 4. News Impact & Decay Engine Tests
# ==========================================

def test_impact_engine_decay_and_buzz():
    engine = NewsImpactEngine(half_life_hours=6.0, baseline_articles_per_day=1.0)
    now = datetime.now(timezone.utc)

    # Article 1: 12 hours ago (decayed)
    art_old = NewsArticle(
        id="art_old",
        headline="Amazon beats earnings expectations with solid cloud growth",
        source="Bloomberg",
        symbols=["AMZN"],
        published_at=now - timedelta(hours=12),
    )
    # Article 2: 1 hour ago (fresh)
    art_new = NewsArticle(
        id="art_new",
        headline="Amazon raises full-year profit outlook on surging retail sales",
        source="Reuters",
        symbols=["AMZN"],
        published_at=now - timedelta(hours=1),
    )
    # Article 3: 30 mins ago
    art_new2 = NewsArticle(
        id="art_new2",
        headline="Amazon upgraded to Strong Buy by analysts citing AWS momentum",
        source="CNBC",
        symbols=["AMZN"],
        published_at=now - timedelta(minutes=30),
    )

    engine.process_article(art_old)
    engine.process_article(art_new)
    engine.process_article(art_new2)

    state = engine.get_symbol_sentiment("AMZN", as_of=now)

    assert state.symbol == "AMZN"
    assert state.article_count_24h == 3
    assert state.buzz_ratio >= 2.5
    assert state.is_buzz_spike is True
    assert state.is_bullish_catalyst is True
    assert state.bullish_catalysts_count >= 2
    assert state.composite_news_score > 30.0


def test_impact_engine_feature_dict_export():
    engine = NewsImpactEngine()
    now = datetime.now(timezone.utc)

    art = NewsArticle(
        id="art_bear",
        headline="Meta warns on revenue and cuts full-year forecast sharply",
        source="Bloomberg",
        symbols=["META"],
        published_at=now,
    )
    engine.process_article(art)

    features = engine.to_feature_dict("META")
    assert "news_sentiment_score" in features
    assert "news_decayed_score" in features
    assert "news_buzz_ratio" in features
    assert "news_composite_score" in features
    assert features["news_sentiment_score"] < 0.0
    assert features["news_bearish_catalyst_count"] >= 1


# ==========================================
# 5. Alpaca News Adapter Tests
# ==========================================

@pytest.mark.anyio
async def test_alpaca_news_adapter_parsing():
    adapter = AlpacaNewsAdapter(api_key="test_key", secret_key="test_secret")

    mock_payload = {
        "news": [
            {
                "id": 12345,
                "headline": "Apple unveils new high performance M4 chips",
                "summary": "Apple announced new hardware lineup today.",
                "author": "John Doe",
                "created_at": "2026-09-08T12:00:00Z",
                "source": "Benzinga",
                "url": "https://example.com/news/12345",
                "symbols": ["AAPL"],
            }
        ]
    }

    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
        articles = await adapter.get_latest_news(symbols=["AAPL"], limit=5)

    assert len(articles) == 1
    assert articles[0].id == "12345"
    assert articles[0].headline == "Apple unveils new high performance M4 chips"
    assert "AAPL" in articles[0].symbols
    assert articles[0].source == "Benzinga"


@pytest.mark.anyio
async def test_alpaca_news_adapter_error_handling():
    adapter = AlpacaNewsAdapter(api_key="test_key", secret_key="test_secret")

    with patch("httpx.AsyncClient.get", side_effect=Exception("Connection timeout")):
        articles = await adapter.get_latest_news(symbols=["AAPL"])

    # Gracefully returns empty list on network failure
    assert articles == []
