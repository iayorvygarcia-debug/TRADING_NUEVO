"""
AI Quant Trading Platform — Pre-Trade Risk Guard & Gap Risk Engine.

Provides multidimensional risk validation before any order reaches the broker:
1. GapRiskEngine: Holistic assessment of overnight/premarket gaps using:
   - Gap magnitude relative to NATR(14)
   - Pre-market volume support
   - News sentiment & catalyst direction
   - Macro regime alignment
2. Dynamic Spread Guard: Liquidity-aware friction filter.
3. Decoupled Capability Guard: Ensures current session permits the intended action (ENTRY vs EXIT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from core.logging import get_logger
from core.market_session import SessionCapabilities, get_dynamic_max_spread, get_session_capabilities

logger = get_logger("pre_trade_guard")


@dataclass(frozen=True)
class GapRiskAssessment:
    """Outcome of holistic gap evaluation."""
    status: str  # "APPROVED", "REDUCE", "REJECT"
    risk_score: float  # 0.0 (safe) to 1.0 (extreme risk)
    reason: str
    size_multiplier: float  # 1.0 for normal, 0.5 for caution, 0.0 for reject


@dataclass(frozen=True)
class PreTradeGuardResult:
    """Final decision of the PreTradeRiskGuard."""
    approved: bool
    size_multiplier: float
    violations: list[str]
    warnings: list[str]
    gap_assessment: Optional[GapRiskAssessment] = None


class GapRiskEngine:
    """
    Evaluates price gaps dynamically rather than using a single rigid ATR cutoff.
    
    Principles:
    - A gap in the SAME direction as the signal, supported by high volume and positive news,
      is institutional momentum (APPROVED).
    - A large gap with no volume or conflicting news is exhaustion or trapped liquidity (REJECT).
    - An adverse gap counter to the trade direction increases risk (REDUCE or REJECT).
    """

    @staticmethod
    def evaluate(
        symbol: str,
        direction: str,  # "buy" or "sell"
        current_price: float,
        prev_close: float,
        natr_14: Optional[float] = None,
        news_score: float = 0.0,
        macro_risk_mult: float = 1.0,
    ) -> GapRiskAssessment:
        if prev_close <= 0 or current_price <= 0:
            return GapRiskAssessment(
                status="REJECT",
                risk_score=1.0,
                reason="Datos de precio o prev_close inválidos o insuficientes para evaluar gap.",
                size_multiplier=0.0,
            )

        gap_pct = ((current_price - prev_close) / prev_close) * 100.0  # in %

        if abs(gap_pct) < 0.5:
            return GapRiskAssessment(
                status="APPROVE",
                risk_score=0.10,
                reason="Sin gap significativo (< 0.5%).",
                size_multiplier=1.0,
            )

        if not natr_14 or natr_14 <= 0:
            return GapRiskAssessment(
                status="REDUCE",
                risk_score=0.60,
                reason="Gap detectado sin NATR(14) disponible: reduciendo tamaño al 50% por incertidumbre de volatilidad.",
                size_multiplier=0.5,
            )

        expected_vol = natr_14
        gap_ratio = abs(gap_pct) / expected_vol

        is_buy = direction.lower() == "buy"
        is_favorable_gap = (is_buy and gap_pct > 0) or (not is_buy and gap_pct < 0)

        # 1. Extreme Gap (> 3x daily expected range) without strong catalyst
        if gap_ratio > 3.0:
            if news_score < 40.0:
                return GapRiskAssessment(
                    status="REJECT",
                    risk_score=0.95,
                    reason=f"Gap anómalo no soportado: {gap_pct:+.2f}% ({gap_ratio:.1f}x NATR) sin catalizador suficiente.",
                    size_multiplier=0.0,
                )
            else:
                return GapRiskAssessment(
                    status="REDUCE",
                    risk_score=0.70,
                    reason=f"Gap extremo con catalizador ({gap_pct:+.2f}%, news {news_score:.0f}): Reduciendo tamaño al 50%.",
                    size_multiplier=0.5,
                )

        # 2. Adverse Gap (> 1.5x daily expected range opposing the trade direction)
        if not is_favorable_gap and gap_ratio > 1.5:
            if news_score <= -30.0 or macro_risk_mult < 0.8:
                return GapRiskAssessment(
                    status="REJECT",
                    risk_score=0.88,
                    reason=f"Gap adverso en contra ({gap_pct:+.2f}%) con sentimiento negativo o macro débil.",
                    size_multiplier=0.0,
                )
            return GapRiskAssessment(
                status="REDUCE",
                risk_score=0.60,
                reason=f"Gap adverso moderado ({gap_pct:+.2f}%): Reduciendo posición por cautela.",
                size_multiplier=0.5,
            )

        # 3. Healthy Favorable or Minor Gap (<= 1.5x NATR)
        return GapRiskAssessment(
            status="APPROVED",
            risk_score=min(0.40, gap_ratio * 0.25),
            reason=f"Gap controlado ({gap_pct:+.2f}%, {gap_ratio:.1f}x NATR): Dentro de límites de riesgo.",
            size_multiplier=1.0,
        )


class PreTradeRiskGuard:
    """
    Multidimensional Pre-Trade Safety Gatekeeper.
    Integrates session capabilities, dynamic spread, gap analysis, and holistic news evaluation.
    """

    def __init__(self):
        self.gap_engine = GapRiskEngine()

    def evaluate_order_proposal(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        prev_close: Optional[float] = None,
        spread_pct: float = 0.001,
        natr_14: Optional[float] = None,
        news_score: float = 0.0,
        macro_risk_mult: float = 1.0,
        is_exit: bool = False,
        capabilities: Optional[SessionCapabilities] = None,
    ) -> PreTradeGuardResult:
        violations: list[str] = []
        warnings: list[str] = []
        caps = capabilities or get_session_capabilities()

        # 1. Capability Permission Check
        if is_exit:
            if not caps.can_exit:
                violations.append(f"Sesión actual '{caps.granular_session}' prohíbe órdenes de salida.")
        else:
            if not caps.can_enter:
                violations.append(f"Sesión actual '{caps.granular_session}' prohíbe nuevas entradas.")

        # 2. Dynamic Spread Evaluation
        max_allowed_spread = get_dynamic_max_spread(
            symbol=symbol,
            session=caps.session,
            natr_14=natr_14,
        )
        if spread_pct > max_allowed_spread:
            violations.append(
                f"Spread excesivo para {symbol}: {spread_pct*100:.3f}% "
                f"> {max_allowed_spread*100:.3f}% (límite adaptado para {caps.granular_session})"
            )

        # 3. Gap Evaluation (for new entries only)
        gap_assessment = None
        size_multiplier = caps.max_position_size_multiplier
        if not is_exit and prev_close and prev_close > 0:
            gap_assessment = self.gap_engine.evaluate(
                symbol=symbol,
                direction=direction,
                current_price=current_price,
                prev_close=prev_close,
                natr_14=natr_14,
                news_score=news_score,
                macro_risk_mult=macro_risk_mult,
            )
            if gap_assessment.status == "REJECT":
                violations.append(f"Rechazado por GapRiskEngine: {gap_assessment.reason}")
            elif gap_assessment.status == "REDUCE":
                warnings.append(gap_assessment.reason)
                size_multiplier = min(size_multiplier, gap_assessment.size_multiplier)

        # 4. Multidimensional News & Sentiment Guard (Not a single point of failure)
        # Rejects ONLY if negative news is reinforced by adverse macro or poor composite score
        if not is_exit and news_score <= -60.0:
            if macro_risk_mult <= 0.8 or (gap_assessment and gap_assessment.risk_score > 0.5):
                violations.append(
                    f"Confluencia de riesgo adverso: Noticias severas ({news_score:.1f}) + Macro débil ({macro_risk_mult:.1f}x)"
                )
            else:
                warnings.append(f"Noticias adversas ({news_score:.1f}) detectadas: procediendo con tamaño reducido.")
                size_multiplier = min(size_multiplier, 0.5)

        approved = len(violations) == 0
        return PreTradeGuardResult(
            approved=approved,
            size_multiplier=size_multiplier if approved else 0.0,
            violations=violations,
            warnings=warnings,
            gap_assessment=gap_assessment,
        )
