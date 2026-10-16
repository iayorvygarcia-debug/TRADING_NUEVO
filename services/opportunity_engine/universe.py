"""
AI Quant Trading Platform — Universe Management.

Manages predefined and dynamic universes of US Equities and ETFs:
- S&P 500 Top Liquid Components
- Nasdaq 100 Tech Leaders
- Core US Sector & Benchmark ETFs
- High-Beta Momentum Leaders
- Sector mappings for sector exposure management
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Sector(str, Enum):
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"
    HEALTHCARE = "healthcare"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    ENERGY = "energy"
    COMMUNICATION = "communication"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    ETF = "etf"
    OTHER = "other"


# ── Predefined Symbol Universes ──────────────────────────────────────

CORE_ETFS: list[str] = [
    "SPY", "QQQ", "IWM", "DIA",          # Major index benchmarks
    "XLK", "XLF", "XLV", "XLE", "XLI",  # Sector ETFs
    "XLY", "XLP", "XLU", "XLB", "XLRE",
    "TLT", "GLD", "SLV", "USO",          # Treasuries & Commodities
]

SP500_TOP_50: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK.B",
    "JPM", "V", "UNH", "LLY", "XOM", "JNJ", "PG", "MA",
    "HD", "COST", "ABBV", "MRK", "WMT", "CVX", "BAC", "KO",
    "PEP", "ACN", "ADBE", "LIN", "MCD", "NFLX", "TMO", "CSCO",
    "ABT", "DIS", "ORCL", "GE", "INTC", "WFC", "IBM", "CAT",
    "QCOM", "TXN", "VZ", "AMAT", "NOW", "HON", "BA", "AXP",
    "COP", "PM",
]

NASDAQ100_TECH: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "COST", "AMD", "QCOM", "NFLX", "ADBE", "CSCO", "AMAT", "NOW",
    "INTU", "TXN", "MU", "PANW", "LRCX", "KLAC", "SNPS", "CDNS",
    "CRWD", "MELI", "PYPL", "ABNB", "MRVL", "FTNT",
]

MOMENTUM_GROWTH: list[str] = [
    "NVDA", "AMD", "TSLA", "META", "SMCI", "ARM", "PLTR", "COIN",
    "HOOD", "UBER", "SHOP", "NET", "DDOG", "MDB", "SNOW", "CELH",
]

# ── Sector Mappings ──────────────────────────────────────────────────

SYMBOL_SECTOR_MAP: dict[str, Sector] = {
    # Tech
    "AAPL": Sector.TECHNOLOGY, "MSFT": Sector.TECHNOLOGY, "NVDA": Sector.TECHNOLOGY,
    "AVGO": Sector.TECHNOLOGY, "AMD": Sector.TECHNOLOGY, "QCOM": Sector.TECHNOLOGY,
    "ADBE": Sector.TECHNOLOGY, "CSCO": Sector.TECHNOLOGY, "AMAT": Sector.TECHNOLOGY,
    "NOW": Sector.TECHNOLOGY, "INTU": Sector.TECHNOLOGY, "TXN": Sector.TECHNOLOGY,
    "MU": Sector.TECHNOLOGY, "PANW": Sector.TECHNOLOGY, "CRM": Sector.TECHNOLOGY,
    "PLTR": Sector.TECHNOLOGY, "NET": Sector.TECHNOLOGY, "CRWD": Sector.TECHNOLOGY,
    "LRCX": Sector.TECHNOLOGY, "KLAC": Sector.TECHNOLOGY, "SNPS": Sector.TECHNOLOGY,
    "CDNS": Sector.TECHNOLOGY, "MRVL": Sector.TECHNOLOGY, "FTNT": Sector.TECHNOLOGY,
    "ORCL": Sector.TECHNOLOGY, "IBM": Sector.TECHNOLOGY, "ACN": Sector.TECHNOLOGY,
    "INTC": Sector.TECHNOLOGY, "SMCI": Sector.TECHNOLOGY, "ARM": Sector.TECHNOLOGY,
    "DDOG": Sector.TECHNOLOGY, "MDB": Sector.TECHNOLOGY, "SNOW": Sector.TECHNOLOGY,
    "SHOP": Sector.TECHNOLOGY,
    "PYPL": Sector.FINANCIALS, "HOOD": Sector.FINANCIALS,
    "ABNB": Sector.CONSUMER_DISCRETIONARY, "MELI": Sector.CONSUMER_DISCRETIONARY,
    "UBER": Sector.INDUSTRIALS,
    # Communication
    "GOOGL": Sector.COMMUNICATION, "META": Sector.COMMUNICATION, "NFLX": Sector.COMMUNICATION,
    "DIS": Sector.COMMUNICATION, "VZ": Sector.COMMUNICATION, "T": Sector.COMMUNICATION,
    # Consumer Discretionary
    "AMZN": Sector.CONSUMER_DISCRETIONARY, "TSLA": Sector.CONSUMER_DISCRETIONARY,
    "HD": Sector.CONSUMER_DISCRETIONARY, "MCD": Sector.CONSUMER_DISCRETIONARY,
    "NKE": Sector.CONSUMER_DISCRETIONARY, "SBUX": Sector.CONSUMER_DISCRETIONARY,
    # Financials
    "JPM": Sector.FINANCIALS, "V": Sector.FINANCIALS, "MA": Sector.FINANCIALS,
    "BAC": Sector.FINANCIALS, "WFC": Sector.FINANCIALS, "GS": Sector.FINANCIALS,
    "MS": Sector.FINANCIALS, "AXP": Sector.FINANCIALS, "COIN": Sector.FINANCIALS,
    # Healthcare
    "UNH": Sector.HEALTHCARE, "LLY": Sector.HEALTHCARE, "JNJ": Sector.HEALTHCARE,
    "ABBV": Sector.HEALTHCARE, "MRK": Sector.HEALTHCARE, "TMO": Sector.HEALTHCARE,
    # Energy
    "XOM": Sector.ENERGY, "CVX": Sector.ENERGY, "COP": Sector.ENERGY, "SLB": Sector.ENERGY,
    # Industrials
    "GE": Sector.INDUSTRIALS, "CAT": Sector.INDUSTRIALS, "BA": Sector.INDUSTRIALS,
    "HON": Sector.INDUSTRIALS, "UNP": Sector.INDUSTRIALS, "UPS": Sector.INDUSTRIALS,
    # Consumer Staples
    "PG": Sector.CONSUMER_STAPLES, "WMT": Sector.CONSUMER_STAPLES, "KO": Sector.CONSUMER_STAPLES,
    "PEP": Sector.CONSUMER_STAPLES, "COST": Sector.CONSUMER_STAPLES,
}


@dataclass(slots=True)
class InstrumentMetadata:
    """Metadata for a tradable instrument."""
    symbol: str
    name: str
    sector: Sector
    asset_type: str = "us_equity"  # "us_equity" or "etf"
    is_active: bool = True
    min_trade_size: float = 1.0


class UniverseManager:
    """
    Manages symbol universe definitions, sector mappings, and universe expansion.
    """

    @classmethod
    def get_universe(cls, universe_name: str) -> list[str]:
        """Get list of symbols for a named universe."""
        u_name = universe_name.lower()
        if u_name in ("sp500", "sp500_top", "sp500_top_50"):
            return SP500_TOP_50.copy()
        elif u_name in ("nasdaq", "nasdaq100", "tech"):
            return NASDAQ100_TECH.copy()
        elif u_name in ("etf", "etfs", "core_etfs"):
            return CORE_ETFS.copy()
        elif u_name in ("momentum", "growth"):
            return MOMENTUM_GROWTH.copy()
        elif u_name in ("all", "master"):
            # De-duplicate union of all universes
            combined = set(SP500_TOP_50 + NASDAQ100_TECH + CORE_ETFS + MOMENTUM_GROWTH)
            return sorted(list(combined))
        else:
            raise ValueError(f"Unknown universe: '{universe_name}'")

    @classmethod
    def get_nasdaq_100_tech(cls) -> list[str]:
        return NASDAQ100_TECH.copy()

    @classmethod
    def get_sp500_top_50(cls) -> list[str]:
        return SP500_TOP_50.copy()

    @classmethod
    def get_sector(cls, symbol: str) -> Sector:
        """Resolve sector for a symbol."""
        sym = symbol.upper()
        if sym in CORE_ETFS:
            return Sector.ETF
        return SYMBOL_SECTOR_MAP.get(sym, Sector.OTHER)
