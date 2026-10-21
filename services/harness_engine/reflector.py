"""Experience Reflector — Forensic failure analysis and trade diagnostics (Harness²)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from services.harness_engine.models import (
    FailureCategory,
    HarnessComponent,
    HarnessDiagnosis,
    HarnessLesson,
)

logger = get_logger("harness.reflector")


class ExperienceReflector:
    """Analyzes trade executions and PnL to extract causal diagnostics and lessons."""

    def __init__(self, operational_capital: float = 1_000.0) -> None:
        self.operational_capital = operational_capital

    def diagnose_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl_usd: float,
        pnl_pct: float,
        strategies: Optional[List[str]] = None,
        holding_seconds: float = 0.0,
        spread_pct: float = 0.001,
        natr_14: float = 0.015,
        news_score: float = 0.0,
        regime: str = "regular",
    ) -> HarnessDiagnosis:
        """Runs multi-factor causal analysis on a completed trade."""
        strategies = strategies or []
        notional = abs(quantity * entry_price)
        diag_id = f"diag_{symbol}_{uuid.uuid4().hex[:8]}"

        # 1. Capital Overexposure Check (Sizing failure)
        # If position notional exceeded 35% of the $1,000 operational capital
        is_overexposed = notional > (self.operational_capital * 0.35)

        # 2. Identify Failure Category
        category = FailureCategory.UNKNOWN
        explanation = ""
        mitigation = ""

        if pnl_usd < 0:
            abs_loss = abs(pnl_usd)
            if is_overexposed:
                category = FailureCategory.SIZING_OVEREXPOSURE
                explanation = (
                    f"Posición de {symbol} sobredimensionada: ${notional:,.2f} USD "
                    f"({(notional / self.operational_capital) * 100:.1f}% del capital operativo de ${self.operational_capital:,.0f}). "
                    f"Una fluctuación menor de {pnl_pct:.2f}% generó una pérdida desproporcionada de -${abs_loss:.2f} USD."
                )
                mitigation = (
                    f"Restringir tamaño máximo de posición en {symbol} a $200 USD (20% de ${self.operational_capital:,.0f}) "
                    f"para que un stop de 2% no supere -$4 USD."
                )
            elif spread_pct > 0.0035 and abs(pnl_pct) < (spread_pct * 100 * 2.0):
                category = FailureCategory.SPREAD_SLIPPAGE_FRICTION
                explanation = (
                    f"Fricción de liquidez en {symbol}: el spread bid-ask ({spread_pct * 100:.2f}%) "
                    f"consumió la mayor parte del margen operativo de la operación."
                )
                mitigation = f"Aumentar el filtro de spread para {symbol} a máximo 0.25% en horario regular."
            elif (side.upper() in {"BUY", "LONG"} and news_score >= 30.0 and pnl_pct < -0.8):
                category = FailureCategory.NEWS_SENTIMENT_TRAP
                explanation = (
                    f"Trampa de sentimiento (Sell-the-News) en {symbol}: Gemini detectó noticias alcistas "
                    f"(score {news_score:+.1f}), pero el precio fue absorbido por venta institucional."
                )
                mitigation = "Inyectar regla en prompt de Gemini: exigir confirmación de volumen y VWAP antes de comprar catalizadores matutinos."
            elif holding_seconds < 300 and abs(pnl_pct) < 1.2 and natr_14 > 0.025:
                category = FailureCategory.PREMATURE_STOP
                explanation = (
                    f"Stop prematuro por ruido de mercado en {symbol}: la volatilidad normal (NATR {natr_14 * 100:.1f}%) "
                    f"tocó el stop loss antes de que la tesis tuviera tiempo de madurar (duración: {holding_seconds / 60:.1f} min)."
                )
                mitigation = "Ampliar distancia de stop loss adaptativo usando 1.8x ATR en lugar de stops porcentuales fijos."
            elif "trend_following" in strategies and pnl_pct < -1.0:
                category = FailureCategory.COUNTER_TREND_WHIPSAW
                explanation = (
                    f"Whipsaw contra-tendencia en {symbol}: la estrategia de tendencia entró en un giro falso "
                    f"dentro de un mercado lateral o en consolidación."
                )
                mitigation = "Aumentar requisito de consenso a mínimo 3 agentes y exigir ADX > 22 para habilitar trend-following."
            else:
                category = FailureCategory.REGIME_MISALIGNMENT
                explanation = (
                    f"Operación fallida en {symbol} ({side}) con pérdida de -${abs_loss:.2f} USD ({pnl_pct:.2f}%). "
                    f"Desalineación entre el patrón de entrada y el flujo de órdenes del sector."
                )
                mitigation = "Reducir ponderación de estrategias participantes durante la siguiente sesión."
        else:
            explanation = (
                f"Operación ganadora en {symbol} ({side}): +${pnl_usd:.2f} USD (+{pnl_pct:.2f}%). "
                f"Las estrategias {', '.join(strategies) if strategies else 'del comité'} capturaron el impulso correctamente."
            )
            mitigation = "Mantener ponderaciones y registrar parámetros de entrada exitosos."

        diagnosis = HarnessDiagnosis(
            id=diag_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_usd=round(pnl_usd, 2),
            pnl_pct=round(pnl_pct, 2),
            holding_seconds=holding_seconds,
            failure_category=category,
            root_cause_explanation=explanation,
            mitigation_action=mitigation,
            strategies_involved=strategies,
            news_score_at_entry=news_score,
            spread_at_entry=spread_pct,
            natr_at_entry=natr_14,
        )

        logger.info(
            "harness_diagnosis_completed",
            symbol=symbol,
            pnl_usd=round(pnl_usd, 2),
            category=category.value,
        )
        return diagnosis

    def extract_lesson(self, diagnosis: HarnessDiagnosis) -> Optional[HarnessLesson]:
        """Distills an actionable, evidence-based lesson from a diagnosis."""
        if diagnosis.pnl_usd >= 0:
            return None

        lesson_id = f"lesson_{diagnosis.symbol}_{uuid.uuid4().hex[:6]}"
        comp = HarnessComponent.GUARDRAILS
        text = ""

        if diagnosis.failure_category == FailureCategory.SIZING_OVEREXPOSURE:
            comp = HarnessComponent.POSITION_SIZING
            text = f"Limitar tamaño nocional a máximo $200 USD por posición en base a capital operativo de ${self.operational_capital:,.0f}."
        elif diagnosis.failure_category == FailureCategory.SPREAD_SLIPPAGE_FRICTION:
            comp = HarnessComponent.GUARDRAILS
            text = f"Bloquear entradas en {diagnosis.symbol} si el spread supera el 0.25% del precio."
        elif diagnosis.failure_category == FailureCategory.NEWS_SENTIMENT_TRAP:
            comp = HarnessComponent.SYSTEM_PROMPT
            text = f"En {diagnosis.symbol}, verificar absorción institucional de volumen antes de comprar noticias con score > 30."
        elif diagnosis.failure_category == FailureCategory.COUNTER_TREND_WHIPSAW:
            comp = HarnessComponent.AGENT_WEIGHTS
            text = "Reducir peso de trend_following en mercados sin tendencia definida (ADX < 20)."
        else:
            comp = HarnessComponent.GUARDRAILS
            text = f"Requerir confirmación de 3 agentes o más antes de operar {diagnosis.symbol}."

        return HarnessLesson(
            id=lesson_id,
            symbol=diagnosis.symbol,
            component=comp,
            lesson_text=text,
            evidence=f"Diagnóstico {diagnosis.id}: pérdida de -${abs(diagnosis.pnl_usd):.2f} USD ({diagnosis.pnl_pct:.2f}%) en {diagnosis.symbol}.",
            confidence=0.85,
        )
