"""Propose-Probe-Compose — Task-Level & Harnessing-Level Adaptation Engine (Harness²)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from core.logging import get_logger
from services.harness_engine.models import (
    FailureCategory,
    HarnessComponent,
    HarnessDiagnosis,
    HarnessLesson,
    HarnessState,
)

logger = get_logger("harness.composer")


class ProposeProbeComposer:
    """Implements the Propose-Probe-Compose loop to evolve agent rules and prompts."""

    def __init__(self, state: HarnessState) -> None:
        self.state = state

    def propose_updates(self, lessons: List[HarnessLesson]) -> Dict[str, Any]:
        """Proposes concrete parameter and prompt edits based on accumulated lessons."""
        proposed_weight_deltas: Dict[str, float] = {}
        proposed_prompt_rules: List[str] = []
        proposed_spread_limits: Dict[str, float] = {}

        for lesson in lessons:
            if not lesson.is_active:
                continue

            if lesson.component == HarnessComponent.AGENT_WEIGHTS:
                # E.g. lower weight of failing strategy by 0.02
                strat = "trend_following"
                proposed_weight_deltas[strat] = proposed_weight_deltas.get(strat, 0.0) - 0.02
            elif lesson.component == HarnessComponent.SYSTEM_PROMPT:
                rule = f"WARNING: {lesson.lesson_text}"
                if rule not in proposed_prompt_rules:
                    proposed_prompt_rules.append(rule)
            elif lesson.component == HarnessComponent.GUARDRAILS and lesson.symbol:
                proposed_spread_limits[lesson.symbol] = 0.0025  # 0.25% hard cap

        return {
            "weights": proposed_weight_deltas,
            "prompt_rules": proposed_prompt_rules,
            "spread_limits": proposed_spread_limits,
        }

    def compose_state(self, proposals: Dict[str, Any]) -> HarnessState:
        """Composes validated updates into the persistent HarnessState."""
        # Update weights
        for strat, delta in proposals.get("weights", {}).items():
            curr = self.state.dynamic_weights_adjustment.get(strat, 0.0)
            # Bound adjustments between -0.05 and +0.05
            self.state.dynamic_weights_adjustment[strat] = round(max(-0.05, min(0.05, curr + delta)), 3)

        # Update prompt additions
        for rule in proposals.get("prompt_rules", []):
            if rule not in self.state.prompt_negative_examples:
                self.state.prompt_negative_examples.append(rule)
                if len(self.state.prompt_negative_examples) > 10:
                    self.state.prompt_negative_examples.pop(0)

        # Update spread limits
        for sym, limit in proposals.get("spread_limits", {}).items():
            self.state.tightened_spread_tickers[sym] = limit

        logger.info(
            "harness_state_composed",
            dynamic_weights=self.state.dynamic_weights_adjustment,
            prompt_rules_count=len(self.state.prompt_negative_examples),
            tightened_tickers=list(self.state.tightened_spread_tickers.keys()),
        )
        return self.state
