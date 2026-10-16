"""High-conviction financial catalyst extractor using domain pattern recognition."""

import re
from typing import List, Tuple
from services.news_engine.models import NewsCatalyst, NewsCatalystType


CATALYST_PATTERNS: List[Tuple[NewsCatalystType, int, float, str]] = [
    # 0. Sell-the-News Divergence (Highest priority check to catch "tops estimates but shares fall")
    (
        NewsCatalystType.SELL_THE_NEWS_DIVERGENCE,
        -1,
        0.92,
        r"\b(tops?|beats?|exceeds?|record)\b.{1,60}\b(but|yet|however|while)\b.{1,50}\b(shares?\s+(fall|drop|slide|slump|dip)|falls?|drops?|slides?|slumps?|warns?|short\s+of\s+(customer\s+)?demand)\b",
    ),
    # 1. Earnings Beat
    (
        NewsCatalystType.EARNINGS_BEAT,
        1,
        0.95,
        r"\b(beats?|surpass(es|ed)?|crush(es|ed)?|top(s|ped)?|exceed(s|ed)?)\s+(on\s+)?(q[1-4]\s+|202[0-9]\s+)?(earnings|eps|revenue|profit|estimates?|forecasts?|consensus|net\s+sales)\b",
    ),
    (
        NewsCatalystType.EARNINGS_BEAT,
        1,
        0.90,
        r"\b(record\s+(quarterly\s+)?(revenue|profit|earnings|net\s+sales|gross\s+margin)|q[1-4]\s+(profit|earnings|net\s+income)\s+(surges?|jumps?|soars?|climbs?))\b",
    ),
    # 2. Earnings Miss
    (
        NewsCatalystType.EARNINGS_MISS,
        -1,
        0.95,
        r"\b(miss(es|ed)?|lag(s|ged)?|fall(s|en)?\s+short\s+of)\s+(on\s+)?(q[1-4]\s+|customer\s+)?(earnings|eps|revenue|profit|estimates?|forecasts?|consensus|demand)\b",
    ),
    (
        NewsCatalystType.EARNINGS_MISS,
        -1,
        0.90,
        r"\b(q[1-4]\s+(loss|misses)|unexpected\s+(loss|drop\s+in\s+profit))\b",
    ),
    # 3. Guidance Raise
    (
        NewsCatalystType.GUIDANCE_RAISE,
        1,
        0.95,
        r"\b(raise(s|d)?|lift(s|ed)?|boost(s|ed)?|increase(s|d)?)\s+(full-?year|fy\d*|q[1-4]|annual)?\s*(guidance|outlook|forecast|capex)\b",
    ),
    (
        NewsCatalystType.GUIDANCE_RAISE,
        1,
        0.90,
        r"\b(sees|anticipates)\s+(higher|stronger|record)\s+(revenue|profit|growth|margins?|sales|ai\s+demand)\b",
    ),
    # 4. Guidance Lower
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
        r"\b(warns?\s+on|warns?\s+of\s+weakness|sees\s+lower|revenue\s+warning|supply\s+to\s+fall\s+short)\b",
    ),
    # 5. AI & Semiconductor Hardware Cycle (2026 Hyperscaler / Silicon Catalysts)
    (
        NewsCatalystType.AI_CHIP_CYCLE,
        1,
        0.88,
        r"\b(unveils?\s+new\s+ai\s+chip|ai\s+demand\s+drives\s+record|agentic\s+ai\s+boosts?|bucks?\s+selloff|hyperscale\s+cloud\s+spending|n2\s+ramp\s+improves|open-?source\s+ai)\b",
    ),
    # 6. Export Controls, Chip Laws & Tariffs
    (
        NewsCatalystType.EXPORT_CONTROL_TARIFF,
        -1,
        0.92,
        r"\b(chip\s+export\s+law|export\s+(ban|controls?|restrictions?)|new\s+chip\s+law|tariff(s)?\s+on|zero\s+china\s+data\s+center\s+revenue|limited\s+ai\s+chip\s+capacity)\b",
    ),
    # 7. Sector Rotation & Pre-Earnings EPS Revisions
    (
        NewsCatalystType.SECTOR_ROTATION,
        -1,
        0.82,
        r"\b(rotation\s+out\s+of|amid\s+chip\s+rout|slide(s)?\s+premarket|drop(s)?\s+[2-9]%\s+as\s+rotation)\b",
    ),
    (
        NewsCatalystType.EARNINGS_REVISION,
        -1,
        0.80,
        r"\b(consensus\s+eps\s+revisions?\s+(fell|dropped|cut|lowered)|downward\s+earnings\s+revisions?)\b",
    ),
    (
        NewsCatalystType.EARNINGS_REVISION,
        1,
        0.80,
        r"\b(pre-?season\s+earnings\s+upgrades?|upward\s+eps\s+revisions?)\b",
    ),
    # 8. FDA Approval & Clinical Trials
    (
        NewsCatalystType.FDA_APPROVAL,
        1,
        1.0,
        r"\b(fda\s+approv(es?|al|ed)|receives?\s+fda\s+clearance|fast-?track\s+designation|phase\s+3\s+success)\b",
    ),
    # 9. FDA Rejection & Clinical Failures
    (
        NewsCatalystType.FDA_REJECTION,
        -1,
        1.0,
        r"\b(fda\s+reject(s?|ion|ed)|complete\s+response\s+letter|crl|clinical\s+hold|phase\s+3\s+fail(ure|s)?)\b",
    ),
    # 10. Mergers & Acquisitions
    (
        NewsCatalystType.MERGER_ACQUISITION,
        1,
        0.85,
        r"\b(to\s+acquire|agrees?\s+to\s+buy|takeover\s+bid|merger\s+agreement|to\s+be\s+acquired\s+by)\b",
    ),
    # 11. Analyst Upgrades
    (
        NewsCatalystType.ANALYST_UPGRADE,
        1,
        0.80,
        r"\b(upgraded?\s+to\s+(buy|outperform|overweight|strong\s+buy)|price\s+target\s+(raised|boosted|hiked))\b",
    ),
    # 12. Analyst Downgrades
    (
        NewsCatalystType.ANALYST_DOWNGRADE,
        -1,
        0.80,
        r"\b(downgraded?\s+to\s+(sell|underperform|underweight|neutral)|price\s+target\s+(cut|slashed|lowered))\b",
    ),
    # 13. Dividend Hike & Buybacks
    (
        NewsCatalystType.DIVIDEND_INCREASE,
        1,
        0.75,
        r"\b(raise(s|d)?|increase(s|d)?|boost(s|ed)?)\s+(quarterly\s+|annual\s+)?dividend|authorizes?\s+\$?[0-9\.]+\s*(billion|million)?\s*buyback\b",
    ),
    # 14. Dividend Cut
    (
        NewsCatalystType.DIVIDEND_CUT,
        -1,
        0.85,
        r"\b(cut(s|ting)?|slash(es|ed)?|suspend(s|ed)?|reduce(s|d)?)\s+(quarterly\s+|annual\s+)?dividend\b",
    ),
    # 15. Legal & Regulatory Probes
    (
        NewsCatalystType.LEGAL_REGULATORY_PROBE,
        -1,
        0.90,
        r"\b(doj|sec|ftc|cfpb)\s+(opens?|launches?|investigates?|probes?|subpoenas?)|antitrust\s+(suit|probe|investigation)|accused\s+of\s+fraud\b",
    ),
    # 16. Patents
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

        # If headline shows relative strength ("bucks selloff", "holds steady"), remove false bearish sector rotation
        if re.search(r"\b(bucks?\s+selloff|holds?\s+steady|defies?\s+rout)\b", combined_text, re.IGNORECASE):
            catalysts = [c for c in catalysts if c.catalyst_type != NewsCatalystType.SECTOR_ROTATION]

        # If SELL_THE_NEWS_DIVERGENCE is triggered, dampen any naive EARNINGS_BEAT weight
        if NewsCatalystType.SELL_THE_NEWS_DIVERGENCE in seen_types:
            for c in catalysts:
                if c.catalyst_type == NewsCatalystType.EARNINGS_BEAT:
                    c.weight = 0.25

        return catalysts
