"""SEC EDGAR Corporate Filings & Insider Intelligence Client."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import httpx
import structlog
from core.config import get_settings
from services.macro_sec.models import (
    FilingType,
    InsiderTransaction,
    SECCompanyFiling,
)

logger = structlog.get_logger(__name__)

# Pre-indexed CIK dictionary for major US tickers to avoid extra network latency
KNOWN_CIK_MAPPING: Dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "META": "0001326801",
    "TSLA": "0001318605",
    "JPM": "0000019617",
    "V": "0001403161",
    "AMD": "0000002488",
    "SPY": "0000884394",
    "QQQ": "0001067839",
}

# Material 8-K Items requiring risk intervention
ADVERSE_8K_ITEMS = {
    "1.03": "Bankruptcy or Receivership",
    "2.04": "Default or Acceleration of Direct Financial Obligation",
    "4.02": "Non-Reliance on Previously Issued Financial Statements (Restatement)",
    "5.02": "Departure of Directors or Principal Officers",
}


class SecEdgarClient:
    """SEC EDGAR integration compliant with SEC rate-limiting and User-Agent policies."""

    def __init__(
        self,
        user_agent: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.user_agent = user_agent or settings.sec.user_agent
        self.base_url = (base_url or settings.sec.base_url).rstrip("/")
        self._headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        }
        self._cik_cache: Dict[str, str] = dict(KNOWN_CIK_MAPPING)

    def get_cik(self, symbol: str) -> Optional[str]:
        """Returns the 10-digit padded CIK string for the given symbol."""
        return self._cik_cache.get(symbol.upper())

    def register_cik(self, symbol: str, cik: str):
        """Registers a symbol-to-CIK mapping."""
        self._cik_cache[symbol.upper()] = cik.zfill(10)

    def parse_filings_payload(self, symbol: str, data: dict) -> List[SECCompanyFiling]:
        """Parses standard SEC EDGAR submissions JSON payload into structured filing models."""
        filings: List[SECCompanyFiling] = []
        cik = str(data.get("cik", "")).zfill(10)
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return filings

        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        forms = recent.get("form", [])
        primary_docs = recent.get("primaryDocument", [])
        items_list = recent.get("items", [])

        for i in range(len(accession_numbers)):
            form_str = forms[i] if i < len(forms) else "OTHER"
            acc_num = accession_numbers[i]
            acc_clean = acc_num.replace("-", "")
            prim_doc = primary_docs[i] if i < len(primary_docs) else ""

            # Classify Form
            if form_str == "10-K":
                f_type = FilingType.FORM_10K
            elif form_str == "10-Q":
                f_type = FilingType.FORM_10Q
            elif form_str == "8-K":
                f_type = FilingType.FORM_8K
            elif form_str == "4":
                f_type = FilingType.FORM_4
            else:
                f_type = FilingType.OTHER

            # Parse 8-K Items
            items_reported: List[str] = []
            is_adverse = False
            if f_type == FilingType.FORM_8K and i < len(items_list):
                raw_item = str(items_list[i])
                if raw_item:
                    items_reported = [it.strip() for it in raw_item.split(",")]
                    for item_code in items_reported:
                        for adv_key in ADVERSE_8K_ITEMS:
                            if adv_key in item_code:
                                is_adverse = True
                                break

            date_str = filing_dates[i] if i < len(filing_dates) else None
            try:
                filing_dt = (
                    datetime.fromisoformat(f"{date_str}T00:00:00+00:00")
                    if date_str
                    else datetime.now(timezone.utc)
                )
            except Exception:
                filing_dt = datetime.now(timezone.utc)

            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{prim_doc}"

            filings.append(
                SECCompanyFiling(
                    cik=cik,
                    symbol=symbol.upper(),
                    filing_type=f_type,
                    filing_date=filing_dt,
                    accession_number=acc_num,
                    primary_doc_url=doc_url,
                    items_reported=items_reported,
                    is_material_adverse_event=is_adverse,
                )
            )

        return filings

    async def get_recent_filings(self, symbol: str) -> List[SECCompanyFiling]:
        """Fetches the latest SEC filings for a symbol via EDGAR API."""
        cik = self.get_cik(symbol)
        if not cik:
            logger.warning("sec_edgar_cik_not_found", symbol=symbol)
            return []

        url = f"{self.base_url}/submissions/CIK{cik}.json"
        try:
            async with httpx.AsyncClient(headers=self._headers, timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return self.parse_filings_payload(symbol, resp.json())
                else:
                    logger.warning("sec_edgar_fetch_status_error", symbol=symbol, status=resp.status_code)
        except Exception as exc:
            logger.error("sec_edgar_network_error", symbol=symbol, error=str(exc))

        return []

    def evaluate_insider_cluster(
        self, transactions: List[InsiderTransaction]
    ) -> Dict[str, float]:
        """Analyzes a list of Form 4 transactions for high-conviction cluster insider buying."""
        if not transactions:
            return {
                "net_insider_flow": 0.0,
                "cluster_buys_count": 0.0,
                "has_ceo_purchase": 0.0,
            }

        total_buy_val = 0.0
        total_sell_val = 0.0
        cluster_buys = 0
        has_ceo_buy = False

        for tx in transactions:
            if tx.transaction_code == "P":
                total_buy_val += tx.total_value
                if tx.is_cluster_worthy_purchase:
                    cluster_buys += 1
                if "CEO" in tx.insider_title.upper() or "CHIEF EXECUTIVE" in tx.insider_title.upper():
                    has_ceo_buy = True
            elif tx.transaction_code == "S":
                total_sell_val += tx.total_value

        net_flow = total_buy_val - total_sell_val

        return {
            "net_insider_flow": round(net_flow, 2),
            "cluster_buys_count": float(cluster_buys),
            "has_ceo_purchase": 1.0 if has_ceo_buy else 0.0,
        }
