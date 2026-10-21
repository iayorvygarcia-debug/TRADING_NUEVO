"""Harness² Engine Coordinator — Persistent memory, trade reflection, and dynamic harness tuning."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from services.harness_engine.models import (
    HarnessDiagnosis,
    HarnessLesson,
    HarnessState,
)
from services.harness_engine.reflector import ExperienceReflector
from services.harness_engine.composer import ProposeProbeComposer

logger = get_logger("harness.engine")

MEMORY_PATH = Path("data/harness_memory.json")


class HarnessEngine:
    """Central Harness² manager coordinating reflection, composition, and memory persistence."""

    def __init__(self, memory_file: Path = MEMORY_PATH) -> None:
        self.memory_file = memory_file
        self.state = self._load_state()
        self.reflector = ExperienceReflector(operational_capital=self.state.operational_capital)
        self.composer = ProposeProbeComposer(self.state)

    def _load_state(self) -> HarnessState:
        """Loads persistent harness memory from disk or creates default state."""
        if self.memory_file.exists():
            try:
                data = json.loads(self.memory_file.read_text(encoding="utf-8"))
                return HarnessState.model_validate(data)
            except Exception as e:
                logger.warning("harness_memory_corrupted_reinitializing", error=str(e))
        return HarnessState(operational_capital=1_000.0)

    def save_state(self) -> None:
        """Persists current harness state and improvement library to disk."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            self.memory_file.write_text(
                self.state.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("harness_memory_save_failed", error=str(e))

    def record_closed_trade(
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
    ) -> HarnessDiagnosis:
        """Reflects on a trade, extracts lessons, and evolves the harness."""
        diagnosis = self.reflector.diagnose_trade(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            strategies=strategies,
            holding_seconds=holding_seconds,
            spread_pct=spread_pct,
            natr_14=natr_14,
            news_score=news_score,
        )

        self.state.diagnoses.append(diagnosis)
        # Keep last 100 diagnoses in memory
        if len(self.state.diagnoses) > 100:
            self.state.diagnoses.pop(0)

        lesson = self.reflector.extract_lesson(diagnosis)
        if lesson:
            self.state.lessons.append(lesson)
            # Run Propose-Probe-Compose on new lesson
            proposals = self.composer.propose_updates([lesson])
            self.composer.compose_state(proposals)

        self.state.last_reflection_time = datetime.now(timezone.utc)
        self.save_state()
        return diagnosis

    def get_adjusted_committee_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Returns baseline weights adapted by Harness² dynamic lessons."""
        adjusted = dict(base_weights)
        for strat, delta in self.state.dynamic_weights_adjustment.items():
            if strat in adjusted:
                adjusted[strat] = max(0.05, min(0.35, adjusted[strat] + delta))
        # Renormalize to sum to 1.0
        tot = sum(adjusted.values())
        if tot > 0:
            adjusted = {k: round(v / tot, 3) for k, v in adjusted.items()}
        return adjusted

    def get_gemini_prompt_additions(self) -> str:
        """Returns evidence-backed guidelines to prepend to Gemini financial news prompt."""
        if not self.state.prompt_negative_examples:
            return ""
        rules = "\n".join(f"- {r}" for r in self.state.prompt_negative_examples[-5:])
        return (
            "\n[HARNESS² EMPIRICAL RULES DERIVED FROM PAST MARKET OUTCOMES]:\n"
            f"{rules}\n"
        )

    def get_effective_max_spread(self, symbol: str, default_max: float) -> float:
        """Returns tightened spread threshold if Harness² identified liquidity friction."""
        return self.state.tightened_spread_tickers.get(symbol, default_max)

    def get_summary(self) -> Dict[str, Any]:
        """Provides an executive status summary for API and dashboard."""
        recent_diag = [d.model_dump() for d in reversed(self.state.diagnoses[-10:])]
        active_lessons = [l.model_dump() for l in self.state.lessons if l.is_active][-10:]
        
        loss_categories: Dict[str, int] = {}
        for d in self.state.diagnoses:
            if d.pnl_usd < 0:
                cat = d.failure_category.value
                loss_categories[cat] = loss_categories.get(cat, 0) + 1

        return {
            "operational_capital": self.state.operational_capital,
            "max_position_size_pct": self.state.max_position_size_pct,
            "total_diagnoses": len(self.state.diagnoses),
            "total_lessons": len(self.state.lessons),
            "loss_distribution_by_cause": loss_categories,
            "dynamic_weights": self.state.dynamic_weights_adjustment,
            "prompt_rules_active": len(self.state.prompt_negative_examples),
            "recent_diagnoses": recent_diag,
            "active_lessons": active_lessons,
            "last_reflection": self.state.last_reflection_time.isoformat() if self.state.last_reflection_time else None,
        }


_GLOBAL_HARNESS_ENGINE: Optional[HarnessEngine] = None


def get_harness_engine() -> HarnessEngine:
    """Singleton getter for the global Harness² engine."""
    global _GLOBAL_HARNESS_ENGINE
    if _GLOBAL_HARNESS_ENGINE is None:
        _GLOBAL_HARNESS_ENGINE = HarnessEngine()
    return _GLOBAL_HARNESS_ENGINE
