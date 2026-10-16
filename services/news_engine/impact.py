"""News Impact, Gemini AI & Sentiment Decay Engine.

Calculates exponential time decay on financial sentiment, tracks news buzz anomalies,
propagates supply-chain sympathy shocks (e.g., TSMC/ASML -> NVDA/AMD/SMCI),
and produces actionable SymbolSentimentState metrics for the 10-Agent Committee.
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional
from services.news_engine.catalyst_extractor import CatalystExtractor
from services.news_engine.dedup import NewsDeduplicator
from services.news_engine.gemini_news_analyzer import (
    GeminiFinancialAnalyzer,
    SUPPLY_CHAIN_SYMPATHY_MAP,
)
from services.news_engine.models import NewsArticle, NewsCatalystType, SymbolSentimentState
from services.news_engine.sentiment import FinancialLexiconAnalyzer


class NewsImpactEngine:
    """Aggregates news, computes time-decayed sentiment, and generates symbol sentiment profiles."""

    def __init__(
        self,
        sentiment_analyzer: Optional[FinancialLexiconAnalyzer] = None,
        catalyst_extractor: Optional[CatalystExtractor] = None,
        deduplicator: Optional[NewsDeduplicator] = None,
        gemini_analyzer: Optional[GeminiFinancialAnalyzer] = None,
        half_life_hours: float = 8.0,
        baseline_articles_per_day: float = 1.0,
        enable_sympathy_propagation: bool = True,
    ):
        self.analyzer = sentiment_analyzer or FinancialLexiconAnalyzer()
        self.extractor = catalyst_extractor or CatalystExtractor()
        self.dedup = deduplicator or NewsDeduplicator()
        self.gemini = gemini_analyzer or GeminiFinancialAnalyzer(
            fallback_analyzer=self.analyzer,
            catalyst_extractor=self.extractor,
        )
        self.half_life_hours = half_life_hours
        self.decay_lambda = math.log(2) / (half_life_hours * 3600.0)
        self.baseline_articles_per_day = baseline_articles_per_day
        self.enable_sympathy_propagation = enable_sympathy_propagation

        # In-memory store of articles by symbol: {symbol: [NewsArticle, ...]}
        self._symbol_articles: Dict[str, List[NewsArticle]] = {}

    def _enrich_article(self, unique_article: NewsArticle, assessment=None) -> NewsArticle:
        """Applies sentiment, catalyst extraction, and sell-the-news detection."""
        full_text = f"{unique_article.headline}. {unique_article.summary or ''}"
        score, conf = self.analyzer.analyze(full_text)
        catalysts = self.extractor.extract(unique_article.headline, unique_article.summary or "")

        if assessment is None:
            assessment = self.gemini.analyze_deterministic_enhanced(
                headline=unique_article.headline,
                summary=unique_article.summary or "",
                symbols=unique_article.symbols,
            )

        # If Sell-The-News divergence is present, use the divergence-corrected score
        if assessment.is_sell_the_news:
            unique_article.sentiment_score = assessment.sentiment_score
            unique_article.sentiment_confidence = assessment.confidence
            unique_article.is_sell_the_news = True
        else:
            # Blend lexicon score with Gemini / enhanced assessment when confidence is higher
            if abs(score) > 0:
                unique_article.sentiment_score = score
                unique_article.sentiment_confidence = conf
            else:
                unique_article.sentiment_score = assessment.sentiment_score
                unique_article.sentiment_confidence = assessment.confidence

        unique_article.recommended_action = assessment.recommended_action
        unique_article.catalysts = catalysts
        return unique_article

    def process_article(self, article: NewsArticle) -> Optional[NewsArticle]:
        """Ingests, enriches, and registers a news article synchronously."""
        unique_article = self.dedup.process_and_deduplicate(article)
        if not unique_article:
            return None

        self._enrich_article(unique_article)

        for sym in unique_article.symbols:
            if sym not in self._symbol_articles:
                self._symbol_articles[sym] = []
            self._symbol_articles[sym].append(unique_article)

        return unique_article

    async def process_article_async(self, article: NewsArticle) -> Optional[NewsArticle]:
        """Ingests and enriches an article asynchronously via Google Gemini AI (with fallback)."""
        unique_article = self.dedup.process_and_deduplicate(article)
        if not unique_article:
            return None

        assessment = await self.gemini.analyze_article_async(unique_article)
        self._enrich_article(unique_article, assessment=assessment)

        for sym in unique_article.symbols:
            if sym not in self._symbol_articles:
                self._symbol_articles[sym] = []
            self._symbol_articles[sym].append(unique_article)

            # Propagate strong supply-chain catalysts (|score| >= 0.65) to sympathy tickers
            if self.enable_sympathy_propagation and abs(unique_article.sentiment_score or 0.0) >= 0.65:
                for related_sym, factor in SUPPLY_CHAIN_SYMPATHY_MAP.get(sym.upper(), []):
                    if related_sym not in unique_article.symbols and not self._symbol_articles.get(related_sym):
                        sympathy_copy = NewsArticle(
                            id=f"{unique_article.id}_sym_{related_sym}",
                            headline=f"[Supply-Chain Spillover from {sym}] {unique_article.headline}",
                            summary=unique_article.summary,
                            source=unique_article.source,
                            url=unique_article.url,
                            symbols=[related_sym],
                            published_at=unique_article.published_at,
                            sentiment_score=round((unique_article.sentiment_score or 0.0) * factor, 4),
                            sentiment_confidence=round((unique_article.sentiment_confidence or 0.5) * factor, 3),
                            source_trust_score=unique_article.source_trust_score,
                            catalysts=[],
                        )
                        self._symbol_articles.setdefault(related_sym, []).append(sympathy_copy)

        return unique_article

    def get_symbol_sentiment(
        self, symbol: str, as_of: Optional[datetime] = None
    ) -> SymbolSentimentState:
        """Calculates current time-decayed sentiment, buzz ratio, and catalyst profile for a symbol."""
        now = as_of or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        all_articles = self._symbol_articles.get(symbol, [])
        direct_articles = [a for a in all_articles if not str(a.id).endswith(f"_sym_{symbol}")]
        articles = direct_articles if direct_articles else all_articles
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

            if hours_old > 72.0:
                continue

            if hours_old <= 24.0:
                count_24h += 1

            time_decay = math.exp(-self.decay_lambda * delta_sec)
            syndication_boost = min(1.35, 1.0 + (max(1, art.syndication_count) - 1) * 0.10)
            weight = (art.sentiment_confidence or 0.5) * art.source_trust_score * time_decay * syndication_boost

            score = art.sentiment_score or 0.0
            total_raw_sentiment += score
            weighted_decayed_sum += score * weight
            total_weights += weight

            if hours_old <= 24.0:
                for cat in art.catalysts:
                    active_catalysts.append(cat)
                    if cat.direction > 0 and not art.is_sell_the_news:
                        bullish_cats += 1
                    elif cat.direction < 0:
                        bearish_cats += 1

        active_count = len([
            a for a in articles
            if (now - (a.published_at if a.published_at.tzinfo else a.published_at.replace(tzinfo=timezone.utc))).total_seconds() <= 72 * 3600
        ])

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

        buzz_ratio = round(count_24h / max(0.5, self.baseline_articles_per_day), 2)

        # Composite score (-100 to +100)
        base_component = decayed_sentiment * 50.0
        net_catalysts = bullish_cats - bearish_cats
        cat_component = max(-30.0, min(30.0, net_catalysts * 15.0))

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
        """Exports key sentiment features for the 10-agent committee and ML models."""
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
