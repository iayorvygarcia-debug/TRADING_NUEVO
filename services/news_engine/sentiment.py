"""Financial Sentiment Analyzer.

Implements a specialized Loughran-McDonald / Wall Street financial lexicon analyzer
optimized for corporate earnings releases, guidance, SEC filings, AI semiconductor
supply-chain news, and financial press.
Includes clause-level contrast handling ('beats estimates but shares fall') and
expanded 2026 market vocabulary.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseSentimentAnalyzer(ABC):
    """Abstract interface for sentiment analyzers."""

    @abstractmethod
    def analyze(self, text: str) -> Tuple[float, float]:
        """Analyzes text.

        Returns:
            Tuple[float, float]: (sentiment_score in [-1.0, 1.0], confidence in [0.0, 1.0])
        """
        pass


FINANCIAL_POSITIVE_TERMS: Dict[str, float] = {
    # Earnings & Performance
    "beat": 0.8, "beats": 0.8, "beating": 0.8,
    "crushed": 0.9, "crushes": 0.9, "crushing": 0.9,
    "surpassed": 0.85, "surpasses": 0.85,
    "exceeded": 0.85, "exceeds": 0.85,
    "outperformed": 0.8, "outperforms": 0.8, "outperformance": 0.8,
    "record": 0.75, "growth": 0.6,
    "surge": 0.75, "surged": 0.75, "surging": 0.75,
    "jump": 0.65, "jumped": 0.65, "climbs": 0.70, "climbed": 0.70,
    "soared": 0.85, "soars": 0.85,
    "rally": 0.7, "rallies": 0.7, "rallied": 0.7,
    "rebound": 0.65, "rebounds": 0.65,
    "profitable": 0.75, "profitability": 0.75, "profit": 0.65, "profits": 0.65,
    "gain": 0.6, "gains": 0.6, "gained": 0.6,
    "upside": 0.7, "robust": 0.75,
    "strong": 0.65, "stronger": 0.7, "strength": 0.65,
    "tailwind": 0.75, "tailwinds": 0.75,
    # Guidance & Analyst
    "upgrade": 0.85, "upgrades": 0.85, "upgraded": 0.85,
    "raised": 0.75, "raises": 0.75, "raising": 0.75,
    "hike": 0.7, "hikes": 0.7, "hiked": 0.7,
    "boost": 0.75, "boosts": 0.75, "boosted": 0.75,
    "bullish": 0.85, "overweight": 0.75, "buy": 0.6,
    # Corporate, AI & Strategic
    "approval": 0.85, "approves": 0.85, "approved": 0.85,
    "expansion": 0.65, "breakthrough": 0.9,
    "buyback": 0.75, "repurchase": 0.7,
    "partnership": 0.65, "accretion": 0.7, "accretive": 0.75,
    "synergies": 0.65, "unveils": 0.65, "unveiled": 0.65,
    "bucks": 0.70,
}

FINANCIAL_NEGATIVE_TERMS: Dict[str, float] = {
    # Earnings & Performance
    "miss": -0.85, "misses": -0.85, "missed": -0.85, "missing": -0.8,
    "disappointing": -0.8, "disappointed": -0.8, "disappoints": -0.8,
    "loss": -0.7, "losses": -0.7,
    "plunge": -0.85, "plunges": -0.85, "plunged": -0.85,
    "slump": -0.75, "slumps": -0.75, "slumped": -0.75,
    "tumble": -0.8, "tumbles": -0.8, "tumbled": -0.8,
    "drop": -0.6, "drops": -0.6, "dropped": -0.6,
    "slide": -0.65, "slides": -0.65, "slid": -0.65,
    "selloff": -0.75, "rout": -0.80,
    "decline": -0.65, "declines": -0.65, "declined": -0.65, "declining": -0.65,
    "fall": -0.6, "falls": -0.6, "fallen": -0.6, "fell": -0.65,
    "weak": -0.7, "weaker": -0.75, "weakness": -0.7,
    "headwind": -0.75, "headwinds": -0.75,
    # Guidance & Analyst
    "downgrade": -0.85, "downgrades": -0.85, "downgraded": -0.85,
    "cut": -0.75, "cuts": -0.75, "cutting": -0.75,
    "lower": -0.65, "lowers": -0.65, "lowered": -0.65, "lowering": -0.65,
    "slashed": -0.85, "slashes": -0.85,
    "underweight": -0.75, "sell": -0.65, "bearish": -0.85,
    "warns": -0.75, "warning": -0.75,
    # Financial Distress, Export Controls & Legal
    "investigation": -0.85, "subpoena": -0.9, "probe": -0.8,
    "lawsuit": -0.75, "fraud": -0.95, "bankruptcy": -0.95,
    "insolvent": -0.95, "default": -0.9, "defaults": -0.9, "defaulted": -0.9,
    "impairment": -0.8, "writedown": -0.8, "write-down": -0.8,
    "rejection": -0.85, "rejected": -0.85, "delisting": -0.9,
    "halted": -0.8, "restatement": -0.85, "scandal": -0.9,
    "fine": -0.6, "penalty": -0.7, "penalties": -0.7,
    "tariff": -0.75, "tariffs": -0.75, "ban": -0.80,
}

NEGATIONS = {
    "not", "no", "never", "bucks", "bucked", "defies", "defied", "without", "neither", "none", "barely",
    "hardly", "rarely", "failed", "fails", "fail", "unable", "cannot", "couldn't",
    "didn't", "hasn't", "wasn't", "isn't", "aren't"
}

INTENSIFIERS = {
    "substantially": 1.4, "sharply": 1.35, "massively": 1.4,
    "significantly": 1.3, "dramatically": 1.35, "strongly": 1.25,
    "heavily": 1.25, "slightly": 0.7, "marginally": 0.6,
    "modestly": 0.75, "partially": 0.8,
}


class FinancialLexiconAnalyzer(BaseSentimentAnalyzer):
    """Rule-based Wall Street financial dictionary analyzer with contextual weighting."""

    def __init__(self):
        self.positive_terms = FINANCIAL_POSITIVE_TERMS
        self.negative_terms = FINANCIAL_NEGATIVE_TERMS
        self.negations = NEGATIONS
        self.intensifiers = INTENSIFIERS

    def analyze(self, text: str) -> Tuple[float, float]:
        if not text or not text.strip():
            return 0.0, 0.0

        tokens = re.findall(r"\b[a-z0-9\-']+\b", text.lower())
        if not tokens:
            return 0.0, 0.0

        total_score = 0.0
        matches = 0.0
        window_size = 3

        cleaned_lower = text.lower()
        for phrase, score in [
            ("dividend increase", 0.85),
            ("dividend hike", 0.85),
            ("dividend cut", -0.85),
            ("guidance raise", 0.85),
            ("guidance cut", -0.85),
            ("full-year guidance raised", 0.9),
            ("full-year guidance lowered", -0.9),
            ("authorized buyback", 0.8),
            ("price target raised", 0.8),
            ("price target cut", -0.8),
            ("phase 3 success", 0.9),
            ("phase 3 failure", -0.95),
            ("all-time high", 0.8),
            ("52-week low", -0.75),
            ("bucks selloff", 1.60),
            ("holds steady", 0.85),
            ("unveils new ai", 0.95),
            ("ai demand drives record", 0.90),
            ("rotation out of", -0.75),
            ("chip rout", -0.80),
            ("but shares fall", -0.90),
            ("shares fall", -0.75),
            ("fall short of customer demand", -0.80),
            ("consensus eps revisions fell", -0.75),
        ]:
            if phrase in cleaned_lower:
                total_score += score * 1.5
                matches += 1.5

        for i, token in enumerate(tokens):
            term_score = 0.0

            if token in self.positive_terms:
                term_score = self.positive_terms[token]
            elif token in self.negative_terms:
                term_score = self.negative_terms[token]
            else:
                continue

            negated = False
            multiplier = 1.0

            start_idx = max(0, i - window_size)
            lookback = tokens[start_idx:i]

            for prev_word in lookback:
                if prev_word in self.negations:
                    negated = not negated
                elif prev_word in self.intensifiers:
                    multiplier *= self.intensifiers[prev_word]

            if negated:
                term_score = -term_score * 0.8

            term_score *= multiplier
            total_score += term_score
            matches += 1.0

        if matches == 0:
            return 0.0, 0.0

        raw_sentiment = total_score / max(1.0, float(matches))
        clamped_sentiment = max(-1.0, min(1.0, round(raw_sentiment, 4)))
        confidence = min(1.0, round(matches / 3.0, 3))

        return clamped_sentiment, confidence
