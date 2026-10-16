"""
Opportunity Engine service package.

Provides universe management, multi-stage screening, 4-factor opportunity scoring,
and ranking engine.
"""

from services.opportunity_engine.universe import (
    Sector,
    UniverseManager,
    SP500_TOP_50,
    NASDAQ100_TECH,
    CORE_ETFS,
    MOMENTUM_GROWTH,
)
from services.opportunity_engine.screener import (
    OpportunityScreener,
    ScreeningResult,
)
from services.opportunity_engine.scorer import (
    OpportunityScorer,
    ScoredOpportunity,
)
from services.opportunity_engine.ranker import (
    UniverseScanner,
)

__all__ = [
    "Sector",
    "UniverseManager",
    "SP500_TOP_50",
    "NASDAQ100_TECH",
    "CORE_ETFS",
    "MOMENTUM_GROWTH",
    "OpportunityScreener",
    "ScreeningResult",
    "OpportunityScorer",
    "ScoredOpportunity",
    "UniverseScanner",
]
