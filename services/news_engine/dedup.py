"""News Deduplication & Source Trust Engine.

Filters duplicate syndicated wire feeds, scores source credibility, and tracks
syndication buzz counts across news releases.
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

    # Tier 2: Established Financial Outlets (0.85)
    "cnbc": 0.85,
    "marketwatch": 0.85,
    "benzinga": 0.85,
    "barron's": 0.85,
    "barrons": 0.85,
    "investor's business daily": 0.85,
    "ibd": 0.85,
    "forbes": 0.80,

    # Tier 3: Crowdsourced / Retail Outlets (0.65)
    "seeking alpha": 0.65,
    "the motley fool": 0.65,
    "motley fool": 0.65,
    "yahoo finance": 0.70,
    "tipranks": 0.65,
    "zacks": 0.65,

    # Tier 4: Social / Unverified (0.40)
    "stocktwits": 0.40,
    "twitter": 0.40,
    "x": 0.40,
    "reddit": 0.35,
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "the", "this", "but", "they",
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
    """Detects syndicated duplicates and aggregates news buzz."""

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
        """Checks article against recent history.

        If it is a duplicate of an existing article:
            - Increments the syndication count of the original article.
            - Returns None (indicating duplicate rejected).
        If unique:
            - Updates source trust score.
            - Adds to cache and returns the enriched article.
        """
        article.source_trust_score = SourceTrustRater.get_trust_score(article.source)
        article_tokens = self._tokenize(article.headline)

        now = article.published_at
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        symbols = article.symbols if article.symbols else ["_ALL_"]

        # Check each symbol for duplicate
        for sym in symbols:
            if sym not in self._history:
                self._history[sym] = []

            # Prune old cache items
            self._history[sym] = [
                a for a in self._history[sym]
                if (now - (a.published_at if a.published_at.tzinfo else a.published_at.replace(tzinfo=timezone.utc))).total_seconds() <= self.max_cache_hours * 3600
            ]

            for existing in self._history[sym]:
                existing_tokens = self._tokenize(existing.headline)
                sim = self._jaccard_similarity(article_tokens, existing_tokens)
                if sim >= self.similarity_threshold:
                    # Found duplicate/syndicated wire
                    existing.syndication_count += 1
                    return None

        # Unique article -> save to symbol caches
        for sym in symbols:
            self._history[sym].append(article)

        return article

    def clear(self):
        """Clears the history cache."""
        self._history.clear()
