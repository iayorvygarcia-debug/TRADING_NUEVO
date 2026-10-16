"""High-conviction financial catalyst extractor using domain pattern recognition."""

import re
from typing import List, Tuple
from services.news_engine.models import NewsCatalyst, NewsCatalystType


CATALYST_PATTERNS: List[Tuple[NewsCatalystType, int, float, str]] = [
    # (Type, Direction, Weight, Regex Pattern)
    # Earnings Beat
    (
        NewsCatalystType.EARNINGS_BEAT,
        1,
        0.95,
        r"\b(beats?|surpass(es|ed)?|crush(es|ed)?|top(s|ped)?|exceed(s|ed)?)\s+(on\s+)?(earnings|eps|revenue|profit|estimates?|forecasts?|consensus)\b",
    ),
    (
        NewsCatalystType.EARNINGS_BEAT,
        1,
        0.90,
        r"\b(record\s+quarterly\s+(revenue|profit|earnings)|q[1-4]\s+(profit|earnings)\s+(surges?|jumps?|soars?))\b",
    ),
    # Earnings Miss
    (
        NewsCatalystType.EARNINGS_MISS,
        -1,
        0.95,
        r"\b(miss(es|ed)?|lag(s|ged)?|fall(s|en)?\s+short\s+of)\s+(on\s+)?(earnings|eps|revenue|profit|estimates?|forecasts?|consensus)\b",
    ),
    (
        NewsCatalystType.EARNINGS_MISS,
        -1,
        0.90,
        r"\b(q[1-4]\s+(loss|misses)|unexpected\s+(loss|drop\s+in\s+profit))\b",
    ),
    # Guidance Raise
    (
        NewsCatalystType.GUIDANCE_RAISE,
        1,
        0.95,
        r"\b(raise(s|d)?|lift(s|ed)?|boost(s|ed)?|increase(s|d)?)\s+(full-?year|fy\d*|q[1-4]|annual)?\s*(guidance|outlook|forecast)\b",
    ),
    (
        NewsCatalystType.GUIDANCE_RAISE,
        1,
        0.90,
        r"\b(sees|anticipates)\s+(higher|stronger)\s+(revenue|profit|growth|margins?|sales)\b",
    ),
    # Guidance Lower
    (
        NewsCatalystType.GUIDANCE_LOWER,
        -1,
        0.95,
        r"\b(cut(s|ting)?|lower(s|ed)?|slash(es|ed)?|reduce(s|d)?)\s+(full-?year|fy\d*|q[1-4]|annual)?\s*(guidance|outlook|forecast)\b",
    ),
    (
        NewsCatalystType.GUIDANCE_LOWER,
        -1,
        0.90,
        r"\b(warns?\s+on|warns?\s+of\s+weakness|sees\s+lower|revenue\s+warning)\b",
    ),
    # FDA Approval & Clinical Trials
    (
        NewsCatalystType.FDA_APPROVAL,
        1,
        1.0,
        r"\b(fda\s+approv(es?|al|ed)|receives?\s+fda\s+clearance|fast-?track\s+designation|phase\s+3\s+success)\b",
    ),
    # FDA Rejection & Clinical Failures
    (
        NewsCatalystType.FDA_REJECTION,
        -1,
        1.0,
        r"\b(fda\s+reject(s?|ion|ed)|complete\s+response\s+letter|crl|clinical\s+hold|phase\s+3\s+fail(ure|s)?)\b",
    ),
    # Mergers & Acquisitions
    (
        NewsCatalystType.MERGER_ACQUISITION,
        1,
        0.85,
        r"\b(to\s+acquire|agrees?\s+to\s+buy|takeover\s+bid|merger\s+agreement|to\s+be\s+acquired\s+by)\b",
    ),
    # Analyst Upgrades
    (
        NewsCatalystType.ANALYST_UPGRADE,
        1,
        0.80,
        r"\b(upgraded?\s+to\s+(buy|outperform|overweight|strong\s+buy)|price\s+target\s+(raised|boosted|hiked))\b",
    ),
    # Analyst Downgrades
    (
        NewsCatalystType.ANALYST_DOWNGRADE,
        -1,
        0.80,
        r"\b(downgraded?\s+to\s+(sell|underperform|underweight|neutral)|price\s+target\s+(cut|slashed|lowered))\b",
    ),
    # Dividend Hike & Buybacks
    (
        NewsCatalystType.DIVIDEND_INCREASE,
        1,
        0.75,
        r"\b(raise(s|d)?|increase(s|d)?|boost(s|ed)?)\s+(quarterly\s+|annual\s+)?dividend|authorizes?\s+\$?[0-9\.]+\s*(billion|million)?\s*buyback\b",
    ),
    # Dividend Cut
    (
        NewsCatalystType.DIVIDEND_CUT,
        -1,
        0.85,
        r"\b(cut(s|ting)?|slash(es|ed)?|suspend(s|ed)?|reduce(s|d)?)\s+(quarterly\s+|annual\s+)?dividend\b",
    ),
    # Legal & Regulatory Probes
    (
        NewsCatalystType.LEGAL_REGULATORY_PROBE,
        -1,
        0.90,
        r"\b(doj|sec|ftc|cfpb)\s+(opens?|launches?|investigates?|probes?|subpoenas?)|antitrust\s+(suit|probe|investigation)|accused\s+of\s+fraud\b",
    ),
    # Patents
    (
        NewsCatalystType.PATENT_INNOVATION,
        1,
        0.70,
        r"\b(awarded|granted|secures?)\s+(key|new|landmark)?\s*patent\b",
    ),
]


class CatalystExtractor:
    """Detects and extracts material catalysts from financial headlines and summaries."""

    def __init__(self):
        self.compiled_patterns = [
            (cat_type, direction, weight, re.compile(pattern_str, re.IGNORECASE))
            for cat_type, direction, weight, pattern_str in CATALYST_PATTERNS
        ]

    def extract(self, headline: str, summary: str = "") -> List[NewsCatalyst]:
        """Extracts catalysts from headline and summary text.

        Returns:
            List[NewsCatalyst]: All matching catalysts identified with direction and confidence weight.
        """
        catalysts: List[NewsCatalyst] = []
        combined_text = f"{headline} {summary}".strip()
        if not combined_text:
            return catalysts

        seen_types = set()

        for cat_type, direction, weight, regex in self.compiled_patterns:
            match = regex.search(combined_text)
            if match and cat_type not in seen_types:
                catalysts.append(
                    NewsCatalyst(
                        catalyst_type=cat_type,
                        direction=direction,
                        weight=weight,
                        headline_snippet=match.group(0),
                    )
                )
                seen_types.add(cat_type)

        return catalysts
