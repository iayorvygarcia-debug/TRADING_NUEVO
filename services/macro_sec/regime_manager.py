"""Market Regime Manager coordinating Macro and SEC EDGAR intelligence."""

from typing import Dict, List, Optional
from services.macro_sec.fred_macro import FredMacroClassifier
from services.macro_sec.models import (
    MacroIndicators,
    MacroRegime,
    SECCompanyFiling,
)
from services.macro_sec.sec_edgar import SecEdgarClient


class MarketRegimeManager:
    """Master coordinator providing portfolio-level macro scaling and company-level risk blocks."""

    def __init__(
        self,
        macro_classifier: Optional[FredMacroClassifier] = None,
        sec_client: Optional[SecEdgarClient] = None,
    ):
        self.macro_classifier = macro_classifier or FredMacroClassifier()
        self.sec_client = sec_client or SecEdgarClient()
        # In-memory adverse symbol set: symbols with recent catastrophic 8-Ks
        self._blocked_symbols: set[str] = set()

    def update_adverse_filings(self, symbol: str, filings: List[SECCompanyFiling]):
        """Evaluates filings and flags symbol if material adverse events are found."""
        for f in filings:
            if f.is_material_adverse_event:
                self._blocked_symbols.add(symbol.upper())
                return
        self._blocked_symbols.discard(symbol.upper())

    def is_symbol_blocked(self, symbol: str) -> bool:
        """True if symbol has unmitigated adverse corporate governance or debt events."""
        return symbol.upper() in self._blocked_symbols

    def get_risk_multiplier(self, symbol: Optional[str] = None) -> float:
        """Returns the final risk scale multiplier [0.0, 1.0].

        If the specific symbol has a material adverse 8-K block, returns 0.0.
        Otherwise returns the global macro risk multiplier.
        """
        if symbol and self.is_symbol_blocked(symbol):
            return 0.0

        indicators = self.macro_classifier.get_current_indicators()
        return indicators.risk_multiplier

    def get_macro_features(self) -> Dict[str, float]:
        """Returns normalized macro features for the Feature Engine."""
        indicators = self.macro_classifier.get_current_indicators()
        return {
            "macro_yield_curve_spread": indicators.yield_curve_10y_2y,
            "macro_vix_level": indicators.vix_level,
            "macro_fed_funds_rate": indicators.fed_funds_rate,
            "macro_risk_multiplier": indicators.risk_multiplier,
            "macro_is_contraction": 1.0 if indicators.regime == MacroRegime.CONTRACTION_RISK_OFF else 0.0,
            "macro_is_expansion": 1.0 if indicators.regime == MacroRegime.EXPANSION_RISK_ON else 0.0,
        }
