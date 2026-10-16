"""News Deduplication & Source Trust Engine.

Filters duplicate syndicated wire feeds, scores source credibility, and tracks
syndication buzz counts across news releases.
Fixed bug: Multi-symbol articles are now evaluated per-symbol without dropping
coverage for secondary tickers when one symbol already saw a related wire.
"""

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from services.news_engine.models import NewsArticle


# Source credibility tiers based on institutional auditing standards
SOURCE_TRUST_RATINGS: Dict[str, float] = {
    # Tier 1: Institutional & Regulatory Wires (1.0)
    "bloomberg": 1.0,
    "reuters": 1.0,
    "sec": 1.0,
    "sec edgar": 1.0,
    "wall street journal": 1.0,
    "wsj": 1.0,
    "dow jones": 1.0,
    "business wire": 1.0,
    "pr newswire": 1.0,
    "globenewswire": 1.0,
    "financial times": 1.0,
    "citadel": 1.0,
    "asml": 1.0,

    # Tier 2: Established Financial Outlets (0.85)
    "cnbc": 0.85,
    "marketwatch": 0.85,
    "benzinga": 0.85,
    "barron's": 0.85,
    "barrons": 0.85,
    "investor's business daily": 0.85,
    "ibd": 0.85,
    "forbes": 0.80,
    "morningstar": 0.85,
    "stonex": 0.85,
    "investing.com": 0.80,
    "geekwire": 0.80,

    # Tier 3: Crowdsourced / Retail Outlets (0.65)
    "seeking alpha": 0.65,
    "the motley fool": 0.65,
    "motley fool": 0.65,
    "yahoo finance": 0.70,
    "tipranks": 0.65,
    "zacks": 0.65,
    "24/7 wall st": 0.70,
    "tradingview": 0.70,
    "tikr": 0.70,

    # Tier 4: Social / Unverified (0.40)
    "stocktwits": 0.40,
    "twitter": 0.40,
    "x": 0.40,
    "reddit": 0.35,
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how"
}


class SourceTrustRater:
    """Evaluates publisher reliability and institutional weight."""

    @staticmethod
    def get_trust_score(source_name: Optional[str]) -> float:
        """Returns trust coefficient [0.1, 1.0] for the given news provider."""
        if not source_name:
            return 0.50
        cleaned = source_name.lower().strip()
        for key, score in SOURCE_TRUST_RATINGS.items():
            if key in cleaned:
                return score
        return 0.50


class NewsDeduplicator:
    """Detects syndicated duplicates and aggregates news buzz per symbol."""

    def __init__(self, similarity_threshold: float = 0.60, max_cache_hours: float = 12.0):
        self.similarity_threshold = similarity_threshold
        self.max_cache_hours = max_cache_hours
        # Cache structured as: {symbol: [NewsArticle, ...]}
        self._history: Dict[str, List[NewsArticle]] = {}

    def _tokenize(self, text: str) -> Set[str]:
        words = re.findall(r"\b[a-z0-9]{3,}\b", text.lower())
        return {w for w in words if w not in STOPWORDS}

    def _jaccard_similarity(self, tokens_a: Set[str], tokens_b: Set[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return intersection / union if union > 0 else 0.0

    def process_and_deduplicate(self, article: NewsArticle) -> Optional[NewsArticle]:
        """Checks article against recent history per symbol.

        Fixes multi-ticker drop bug: only filters out symbols where the article
        was already registered, preserving novel symbols on multi-ticker wires.
        """
        article.source_trust_score = SourceTrustRater.get_trust_score(article.source)
        article_tokens = self._tokenize(article.headline)

        now = article.published_at
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        symbols = list(article.symbols) if article.symbols else ["_ALL_"]
        novel_symbols: List[str] = []

        for sym in symbols:
            if sym not in self._history:
                self._history[sym] = []

            # Prune old cache items
            self._history[sym] = [
                a for a in self._history[sym]
                if (now - (a.published_at if a.published_at.tzinfo else a.published_at.replace(tzinfo=timezone.utc))).total_seconds() <= self.max_cache_hours * 3600
            ]

            is_dup_for_sym = False
            for existing in self._history[sym]:
                existing_tokens = self._tokenize(existing.headline)
                sim = self._jaccard_similarity(article_tokens, existing_tokens)
                if sim >= self.similarity_threshold:
                    existing.syndication_count += 1
                    is_dup_for_sym = True
                    break

            if not is_dup_for_sym:
                novel_symbols.append(sym)
                self._history[sym].append(article)

        if not novel_symbols:
            return None

        if article.symbols:
            article.symbols = novel_symbols
        return article

    def clear(self):
        """Clears the history cache."""
        self._history.clear()
