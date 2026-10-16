"""Google Gemini AI Financial News & Catalyst Intelligence Analyzer.

Connects TRADING_NUEVO directly to Google Gemini (gemini-2.5-flash) to analyze
real-time market headlines, earnings releases, SEC 8-K events, semiconductor/AI
supply-chain spillover, and 'sell-the-news' divergences.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from core.config import get_settings
from services.news_engine.catalyst_extractor import CatalystExtractor
from services.news_engine.models import NewsArticle, NewsCatalyst, NewsCatalystType
from services.news_engine.sentiment import BaseSentimentAnalyzer, FinancialLexiconAnalyzer


class GeminiNewsAssessment(BaseModel):
    """Structured output returned by Gemini for a financial news item."""

    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Directional financial impact score from -1.0 (extremely bearish) to +1.0 (extremely bullish)",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence in the financial impact assessment (0.0 to 1.0)",
    )
    primary_catalyst: str = Field(
        default="GENERAL_FINANCIAL",
        description="Primary catalyst category",
    )
    is_sell_the_news: bool = Field(
        default=False,
        description="True if headline reports positive numbers but price action or guidance is negative",
    )
    recommended_action: str = Field(
        default="HOLD",
        description="Trading bias: STRONG_BUY, BUY, HOLD, SELL, or STRONG_SELL",
    )
    sympathy_tickers: List[str] = Field(
        default_factory=list,
        description="Related tickers impacted via supply chain or sector rotation",
    )
    reasoning: str = Field(
        default="",
        description="Concise quantitative explanation of the catalyst and expected price impact",
    )


SUPPLY_CHAIN_SYMPATHY_MAP: Dict[str, List[Tuple[str, float]]] = {
    "TSM": [("NVDA", 0.70), ("AMD", 0.65), ("AVGO", 0.55), ("AAPL", 0.45), ("AMAT", 0.50)],
    "ASML": [("AMAT", 0.75), ("LRCX", 0.75), ("KLAC", 0.70), ("NVDA", 0.50), ("MU", 0.50)],
    "NVDA": [("AMD", 0.60), ("SMCI", 0.75), ("AVGO", 0.60), ("MU", 0.55), ("ARM", 0.55)],
    "AMD": [("NVDA", 0.45), ("INTC", 0.55), ("MU", 0.45)],
    "MSFT": [("GOOGL", 0.45), ("AMZN", 0.45), ("NOW", 0.40), ("ORCL", 0.40)],
    "GOOGL": [("META", 0.55), ("MSFT", 0.45), ("AMZN", 0.40)],
    "META": [("GOOGL", 0.55), ("PINS", 0.45), ("SNAP", 0.45)],
}


class GeminiFinancialAnalyzer(BaseSentimentAnalyzer):
    """Hybrid Gemini AI + Quantitative Lexicon financial news analyzer."""

    SYSTEM_INSTRUCTION = (
        "You are an institutional quantitative trading news analyst for US Equities (NASDAQ-100 & S&P 500). "
        "Evaluate the headline and summary for immediate intraday and swing price impact. "
        "Pay special attention to: "
        "(1) 'Sell-the-news' divergences (e.g., 'beats Q3 estimates but shares fall on guidance/supply warning' -> negative score), "
        "(2) AI infrastructure & semiconductor supply-chain bottlenecks (TSMC capacity, ASML bookings, custom AI silicon), "
        "(3) Export controls, tariffs, and regulatory probes (DOJ/SEC/Taiwan chip laws), "
        "(4) Pre-earnings consensus EPS revisions and sector rotation. "
        "Respond strictly with valid JSON matching the requested schema."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_analyzer: Optional[FinancialLexiconAnalyzer] = None,
        catalyst_extractor: Optional[CatalystExtractor] = None,
        timeout_seconds: float = 6.0,
    ) -> None:
        settings = get_settings()
        gemini_cfg = getattr(settings, "gemini", None)
        cfg_key = getattr(gemini_cfg, "api_key", "") if gemini_cfg else ""
        if hasattr(cfg_key, "get_secret_value"):
            cfg_key = cfg_key.get_secret_value()

        self.api_key = (
            api_key
            or str(cfg_key)
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        ).strip()
        self.model_name = (
            model_name
            or (getattr(gemini_cfg, "model_name", "") if gemini_cfg else "")
            or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        )
        self.fallback = fallback_analyzer or FinancialLexiconAnalyzer()
        self.extractor = catalyst_extractor or CatalystExtractor()
        self.timeout_seconds = timeout_seconds
        self._cache: Dict[str, GeminiNewsAssessment] = {}

    @property
    def is_gemini_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 10)

    def analyze(self, text: str) -> Tuple[float, float]:
        if not text or not text.strip():
            return 0.0, 0.0
        cached = self._cache.get(text.strip()[:250])
        if cached is not None:
            return cached.sentiment_score, cached.confidence
        return self.fallback.analyze(text)

    def analyze_deterministic_enhanced(
        self, headline: str, summary: str = "", symbols: Optional[List[str]] = None
    ) -> GeminiNewsAssessment:
        full_text = f"{headline}. {summary or ''}".strip()
        score, conf = self.fallback.analyze(full_text)
        catalysts = self.extractor.extract(headline, summary or "")

        lower_text = full_text.lower()
        sell_the_news_patterns = [
            r"\b(tops?|beats?|exceeds?|record)\b.{1,60}\b(but|yet|however|while)\b.{1,50}\b(shares?\s+(fall|drop|slide|slump|dip)|falls?|drops?|slides?|slumps?|warns?|short\s+of\s+(customer\s+)?demand)\b",
        ]
        is_sell_the_news = any(re.search(p, lower_text) for p in sell_the_news_patterns)
        if is_sell_the_news and score > -0.15:
            score = round(min(-0.35, score - 0.65), 4)
            conf = max(conf, 0.78)

        primary_cat = "GENERAL_FINANCIAL"
        if catalysts:
            strongest = max(catalysts, key=lambda c: c.weight)
            primary_cat = strongest.catalyst_type.value
            if not is_sell_the_news:
                cat_boost = strongest.direction * strongest.weight * 0.35
                score = max(-1.0, min(1.0, round(score + cat_boost, 4)))
                conf = max(conf, round(strongest.weight, 3))
        elif is_sell_the_news:
            primary_cat = NewsCatalystType.SELL_THE_NEWS_DIVERGENCE.value

        if score >= 0.55:
            action = "STRONG_BUY"
        elif score >= 0.20:
            action = "BUY"
        elif score <= -0.55:
            action = "STRONG_SELL"
        elif score <= -0.20:
            action = "SELL"
        else:
            action = "HOLD"

        sympathy: List[str] = []
        for sym in symbols or []:
            for related_sym, _ in SUPPLY_CHAIN_SYMPATHY_MAP.get(sym.upper(), []):
                if related_sym not in (symbols or []) and related_sym not in sympathy:
                    sympathy.append(related_sym)

        assessment = GeminiNewsAssessment(
            sentiment_score=score,
            confidence=conf,
            primary_catalyst=primary_cat,
            is_sell_the_news=is_sell_the_news,
            recommended_action=action,
            sympathy_tickers=sympathy[:5],
            reasoning=f"Catalyst={primary_cat}, score={score:+.2f}, sell_the_news={is_sell_the_news}",
        )
        self._cache[full_text[:250]] = assessment
        return assessment

    async def analyze_article_async(self, article: NewsArticle) -> GeminiNewsAssessment:
        cache_key = f"{article.headline}. {article.summary or ''}".strip()[:250]
        if cache_key in self._cache:
            return self._cache[cache_key]

        baseline = self.analyze_deterministic_enhanced(
            headline=article.headline,
            summary=article.summary or "",
            symbols=article.symbols,
        )

        if not self.is_gemini_configured:
            return baseline

        prompt = (
            f"Analyze this financial news article for {', '.join(article.symbols) or 'US Market'}:\n"
            f"Headline: {article.headline}\n"
            f"Summary: {article.summary or 'N/A'}\n"
            f"Source: {article.source}\n\n"
            "Return JSON with fields: sentiment_score (float -1.0 to 1.0), confidence (float 0.0 to 1.0), "
            "primary_catalyst (string), is_sell_the_news (boolean), recommended_action "
            "(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL), sympathy_tickers (array of ticker strings), "
            "reasoning (short string)."
        )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        payload: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": self.SYSTEM_INSTRUCTION}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            parsed_json = json.loads(parts[0]["text"])
                            assessment = GeminiNewsAssessment(
                                sentiment_score=max(-1.0, min(1.0, float(parsed_json.get("sentiment_score", baseline.sentiment_score)))),
                                confidence=max(0.0, min(1.0, float(parsed_json.get("confidence", baseline.confidence)))),
                                primary_catalyst=str(parsed_json.get("primary_catalyst", baseline.primary_catalyst)),
                                is_sell_the_news=bool(parsed_json.get("is_sell_the_news", baseline.is_sell_the_news)),
                                recommended_action=str(parsed_json.get("recommended_action", baseline.recommended_action)),
                                sympathy_tickers=list(parsed_json.get("sympathy_tickers", baseline.sympathy_tickers)),
                                reasoning=str(parsed_json.get("reasoning", baseline.reasoning)),
                            )
                            self._cache[cache_key] = assessment
                            return assessment
        except Exception:
            pass

        return baseline