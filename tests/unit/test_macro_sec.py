"""Unit tests for Macro & SEC EDGAR Intelligence Engine."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from services.macro_sec.fred_macro import FredMacroClassifier
from services.macro_sec.models import (
    FilingType,
    InsiderTransaction,
    MacroRegime,
    SECCompanyFiling,
)
from services.macro_sec.regime_manager import MarketRegimeManager
from services.macro_sec.sec_edgar import SecEdgarClient


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ==========================================
# 1. FRED Macro Classifier Tests
# ==========================================

def test_macro_regime_expansion():
    classifier = FredMacroClassifier()
    # Normal yield curve (+0.50%), low VIX (14.2), moderate Fed rate (3.5%)
    ind = classifier.classify_regime(
        yield_curve_10y_2y=0.50,
        fed_funds_rate=3.50,
        vix_level=14.2,
    )
    assert ind.regime == MacroRegime.EXPANSION_RISK_ON
    assert ind.risk_multiplier == 1.00
    assert ind.is_low_volatility is True
    assert ind.is_yield_curve_inverted is False


def test_macro_regime_contraction_high_vix():
    classifier = FredMacroClassifier()
    # High VIX spike (32.5) -> immediate contraction
    ind = classifier.classify_regime(
        yield_curve_10y_2y=0.10,
        fed_funds_rate=4.50,
        vix_level=32.5,
    )
    assert ind.regime == MacroRegime.CONTRACTION_RISK_OFF
    assert ind.risk_multiplier == 0.40
    assert ind.is_high_volatility is True


def test_macro_regime_contraction_deep_inversion():
    classifier = FredMacroClassifier()
    # Inverted curve (-0.55%) with high Fed rate (5.25%)
    ind = classifier.classify_regime(
        yield_curve_10y_2y=-0.55,
        fed_funds_rate=5.25,
        vix_level=20.0,
    )
    assert ind.regime == MacroRegime.CONTRACTION_RISK_OFF
    assert ind.risk_multiplier == 0.40


def test_macro_regime_late_cycle_defensive():
    classifier = FredMacroClassifier()
    # Slightly inverted curve (-0.10%), moderate VIX (21.0)
    ind = classifier.classify_regime(
        yield_curve_10y_2y=-0.10,
        fed_funds_rate=4.00,
        vix_level=21.0,
    )
    assert ind.regime == MacroRegime.LATE_CYCLE_DEFENSIVE
    assert ind.risk_multiplier == 0.70


@pytest.mark.anyio
async def test_fred_macro_fetch_mock():
    classifier = FredMacroClassifier(api_key="test_fred_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2026-09-08", "value": "0.35"},
            {"date": "2026-09-07", "value": "0.32"},
        ]
    }

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
        val = await classifier.fetch_series_latest_value("T10Y2Y")

    assert val == 0.35


# ==========================================
# 2. SEC EDGAR Client & Parsing Tests
# ==========================================

def test_sec_edgar_cik_lookup():
    client = SecEdgarClient()
    assert client.get_cik("AAPL") == "0000320193"
    assert client.get_cik("NVDA") == "0001045810"
    
    # Custom registration
    client.register_cik("TEST", "9999")
    assert client.get_cik("TEST") == "0000009999"


def test_sec_edgar_payload_parsing_and_adverse_8k():
    client = SecEdgarClient()
    mock_data = {
        "cik": "320193",
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000106", "0000320193-24-000107"],
                "filingDate": ["2026-08-01", "2026-08-05"],
                "form": ["10-Q", "8-K"],
                "primaryDocument": ["aapl-20260629.htm", "aapl-8k.htm"],
                "items": ["", "4.02 Non-Reliance on Financials"],
            }
        }
    }

    filings = client.parse_filings_payload("AAPL", mock_data)
    assert len(filings) == 2

    # Filing 1: 10-Q
    assert filings[0].filing_type == FilingType.FORM_10Q
    assert filings[0].is_material_adverse_event is False

    # Filing 2: 8-K with Item 4.02 restatement -> adverse!
    assert filings[1].filing_type == FilingType.FORM_8K
    assert filings[1].is_material_adverse_event is True
    assert "4.02" in filings[1].items_reported[0]


def test_sec_insider_cluster_evaluation():
    client = SecEdgarClient()
    now = datetime.now(timezone.utc)

    # CEO purchases $250,000 in open market
    tx1 = InsiderTransaction(
        symbol="AAPL",
        insider_name="Tim Cook",
        insider_title="Chief Executive Officer",
        transaction_code="P",
        shares=1000.0,
        price_per_share=250.0,
        total_value=250000.0,
        transaction_date=now,
        filing_date=now,
    )
    # Director purchases $100,000 in open market
    tx2 = InsiderTransaction(
        symbol="AAPL",
        insider_name="Arthur Levinson",
        insider_title="Director",
        transaction_code="P",
        shares=400.0,
        price_per_share=250.0,
        total_value=100000.0,
        transaction_date=now,
        filing_date=now,
    )
    # Executive routine option sale
    tx3 = InsiderTransaction(
        symbol="AAPL",
        insider_name="Luca Maestri",
        insider_title="CFO",
        transaction_code="S",
        shares=200.0,
        price_per_share=250.0,
        total_value=50000.0,
        transaction_date=now,
        filing_date=now,
    )

    results = client.evaluate_insider_cluster([tx1, tx2, tx3])
    assert results["cluster_buys_count"] == 2.0
    assert results["has_ceo_purchase"] == 1.0
    assert results["net_insider_flow"] == 300000.0  # (250k + 100k) - 50k


# ==========================================
# 3. Market Regime Manager Integration Tests
# ==========================================

def test_regime_manager_risk_multiplier_and_blocks():
    classifier = FredMacroClassifier()
    # Establish expansion baseline
    classifier.classify_regime(0.50, 3.50, 15.0)

    manager = MarketRegimeManager(macro_classifier=classifier)

    # Clean symbol has full risk multiplier
    assert manager.get_risk_multiplier("NVDA") == 1.00

    # Report adverse 8-K for TSLA (e.g. restatement)
    adverse_filing = SECCompanyFiling(
        cik="0001318605",
        symbol="TSLA",
        filing_type=FilingType.FORM_8K,
        filing_date=datetime.now(timezone.utc),
        accession_number="0001318605-26-000010",
        primary_doc_url="https://sec.gov",
        items_reported=["Item 4.02 Restatement"],
        is_material_adverse_event=True,
    )
    manager.update_adverse_filings("TSLA", [adverse_filing])

    # TSLA should be blocked (multiplier 0.0)
    assert manager.is_symbol_blocked("TSLA") is True
    assert manager.get_risk_multiplier("TSLA") == 0.0

    # Other symbols still get global macro multiplier
    assert manager.get_risk_multiplier("AAPL") == 1.00


def test_regime_manager_macro_features():
    classifier = FredMacroClassifier()
    classifier.classify_regime(0.40, 4.00, 16.0)
    manager = MarketRegimeManager(macro_classifier=classifier)

    features = manager.get_macro_features()
    assert "macro_yield_curve_spread" in features
    assert "macro_vix_level" in features
    assert "macro_fed_funds_rate" in features
    assert "macro_risk_multiplier" in features
    assert features["macro_risk_multiplier"] == 1.00
    assert features["macro_is_expansion"] == 1.0
    assert features["macro_is_contraction"] == 0.0
