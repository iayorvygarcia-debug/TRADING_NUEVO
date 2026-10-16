"""News Impact & Sentiment Decay Engine.

Calculates exponential time decay on financial sentiment, tracks news buzz anomalies,
and produces actionable SymbolSentimentState metrics for the Feature & Opportunity Engines.
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional
from services.news_engine.catalyst_extractor import CatalystExtractor
from services.news_engine.dedup import NewsDeduplicator
from services.news_engine.models import NewsArticle, SymbolSentimentState
from services.news_engine.sentiment import FinancialLexiconAnalyzer


class NewsImpactEngine:
    """Aggregates news, computes time-decayed sentiment, and generates symbol sentiment profiles."""

    def __init__(
        self,
        sentiment_analyzer: Optional[FinancialLexiconAnalyzer] = None,
        catalyst_extractor: Optional[CatalystExtractor] = None,
        deduplicator: Optional[NewsDeduplicator] = None,
        half_life_hours: float = 8.0,
        baseline_articles_per_day: float = 1.0,
    ):
        self.analyzer = sentiment_analyzer or FinancialLexiconAnalyzer()
        self.extractor = catalyst_extractor or CatalystExtractor()
        self.dedup = deduplicator or NewsDeduplicator()
        self.half_life_hours = half_life_hours
        # Decay lambda: ln(2) / half_life_seconds
        self.decay_lambda = math.log(2) / (half_life_hours * 3600.0)
        self.baseline_articles_per_day = baseline_articles_per_day

        # In-memory store of articles by symbol: {symbol: [NewsArticle, ...]}
        self._symbol_articles: Dict[str, List[NewsArticle]] = {}

    def process_article(self, article: NewsArticle) -> Optional[NewsArticle]:
        """Ingests, enriches, and registers a news article.

        If duplicate, records syndication and returns None.
        If new, scores sentiment and catalysts, then saves to state.
        """
        # 1. Deduplication & Source Trust Scoring
        unique_article = self.dedup.process_and_deduplicate(article)
        if not unique_article:
            return None

        # 2. Financial Sentiment Analysis
        full_text = f"{unique_article.headline}. {unique_article.summary or ''}"
        score, conf = self.analyzer.analyze(full_text)
        unique_article.sentiment_score = score
        unique_article.sentiment_confidence = conf

        # 3. Catalyst Extraction
        unique_article.catalysts = self.extractor.extract(
            unique_article.headline, unique_article.summary or ""
        )

        # 4. Save to symbol buffers
        for sym in unique_article.symbols:
            if sym not in self._symbol_articles:
                self._symbol_articles[sym] = []
            self._symbol_articles[sym].append(unique_article)

        return unique_article

    def get_symbol_sentiment(
        self, symbol: str, as_of: Optional[datetime] = None
    ) -> SymbolSentimentState:
        """Calculates current time-decayed sentiment, buzz ratio, and catalyst profile for a symbol."""
        now = as_of or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        articles = self._symbol_articles.get(symbol, [])
        if not articles:
            return SymbolSentimentState(
                symbol=symbol,
                raw_sentiment=0.0,
                decayed_sentiment=0.0,
                article_count_24h=0,
                buzz_ratio=1.0,
                bullish_catalysts_count=0,
                bearish_catalysts_count=0,
                active_catalysts=[],
                composite_news_score=0.0,
                last_updated=now,
            )

        total_raw_sentiment = 0.0
        weighted_decayed_sum = 0.0
        total_weights = 0.0
        count_24h = 0
        bullish_cats = 0
        bearish_cats = 0
        active_catalysts = []

        for art in articles:
            art_dt = art.published_at
            if art_dt.tzinfo is None:
                art_dt = art_dt.replace(tzinfo=timezone.utc)

            delta_sec = max(0.0, (now - art_dt).total_seconds())
            hours_old = delta_sec / 3600.0

            # Filter articles older than 72 hours
            if hours_old > 72.0:
                continue

            if hours_old <= 24.0:
                count_24h += 1

            # Time decay factor: e^(-lambda * dt)
            time_decay = math.exp(-self.decay_lambda * delta_sec)
            weight = (art.sentiment_confidence or 0.5) * art.source_trust_score * time_decay

            score = art.sentiment_score or 0.0
            total_raw_sentiment += score
            weighted_decayed_sum += score * weight
            total_weights += weight

            # Catalysts within 24 hours
            if hours_old <= 24.0:
                for cat in art.catalysts:
                    active_catalysts.append(cat)
                    if cat.direction > 0:
                        bullish_cats += 1
                    elif cat.direction < 0:
                        bearish_cats += 1

        active_count = len([a for a in articles if (now - (a.published_at if a.published_at.tzinfo else a.published_at.replace(tzinfo=timezone.utc))).total_seconds() <= 72 * 3600])

        raw_sentiment = (
            round(total_raw_sentiment / max(1, active_count), 4)
            if active_count > 0
            else 0.0
        )
        decayed_sentiment = (
            round(weighted_decayed_sum / max(1e-6, total_weights), 4)
            if total_weights > 0
            else 0.0
        )

        # Buzz ratio: articles in last 24h vs baseline
        buzz_ratio = round(count_24h / max(0.5, self.baseline_articles_per_day), 2)

        # Composite score (-100 to +100)
        # 1. Base from decayed sentiment: [-50, +50]
        base_component = decayed_sentiment * 50.0

        # 2. Catalyst component: each net catalyst adds +/- 15 points (capped at +/- 30)
        net_catalysts = bullish_cats - bearish_cats
        cat_component = max(-30.0, min(30.0, net_catalysts * 15.0))

        # 3. Buzz amplification if aligned
        buzz_multiplier = 1.0
        if buzz_ratio >= 2.0:
            buzz_multiplier = min(1.5, 1.0 + (buzz_ratio - 1.0) * 0.15)

        composite = (base_component + cat_component) * buzz_multiplier
        clamped_composite = max(-100.0, min(100.0, round(composite, 2)))

        return SymbolSentimentState(
            symbol=symbol,
            raw_sentiment=raw_sentiment,
            decayed_sentiment=decayed_sentiment,
            article_count_24h=count_24h,
            buzz_ratio=buzz_ratio,
            bullish_catalysts_count=bullish_cats,
            bearish_catalysts_count=bearish_cats,
            active_catalysts=active_catalysts,
            composite_news_score=clamped_composite,
            last_updated=now,
        )

    def to_feature_dict(self, symbol: str) -> Dict[str, float]:
        """Exports key sentiment features for quantitative models and strategies."""
        state = self.get_symbol_sentiment(symbol)
        return {
            "news_sentiment_score": state.raw_sentiment,
            "news_decayed_score": state.decayed_sentiment,
            "news_buzz_ratio": state.buzz_ratio,
            "news_composite_score": state.composite_news_score,
            "news_bullish_catalyst_count": float(state.bullish_catalysts_count),
            "news_bearish_catalyst_count": float(state.bearish_catalysts_count),
            "news_article_count_24h": float(state.article_count_24h),
        }

    def clear(self):
        """Resets all symbol news buffers."""
        self._symbol_articles.clear()
        self.dedup.clear()
