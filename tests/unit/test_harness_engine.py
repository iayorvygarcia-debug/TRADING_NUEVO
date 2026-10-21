"""Unit tests for Harness² Recursive Agent Harnessing Engine."""

import pytest
from pathlib import Path
from services.harness_engine.models import (
    FailureCategory,
    HarnessComponent,
    HarnessDiagnosis,
    HarnessLesson,
    HarnessState,
)
from services.harness_engine.reflector import ExperienceReflector
from services.harness_engine.composer import ProposeProbeComposer
from services.harness_engine.harness import HarnessEngine


def test_experience_reflector_sizing_overexposure():
    reflector = ExperienceReflector(operational_capital=1000.0)
    # Entry with 14 AAPL @ $344 = $4,816 (> 35% of $1,000)
    diag = reflector.diagnose_trade(
        symbol="AAPL",
        side="BUY",
        entry_price=344.0,
        exit_price=340.0,
        quantity=14.0,
        pnl_usd=-56.0,
        pnl_pct=-1.16,
        strategies=["trend_following"],
    )
    assert diag.failure_category == FailureCategory.SIZING_OVEREXPOSURE
    assert "sobredimensionada" in diag.root_cause_explanation
    lesson = reflector.extract_lesson(diag)
    assert lesson is not None
    assert lesson.component == HarnessComponent.POSITION_SIZING


def test_experience_reflector_spread_friction():
    reflector = ExperienceReflector(operational_capital=1000.0)
    diag = reflector.diagnose_trade(
        symbol="NOW",
        side="BUY",
        entry_price=100.0,
        exit_price=99.5,
        quantity=1.0,
        pnl_usd=-0.50,
        pnl_pct=-0.50,
        spread_pct=0.005,  # 0.5% spread
    )
    assert diag.failure_category == FailureCategory.SPREAD_SLIPPAGE_FRICTION
    lesson = reflector.extract_lesson(diag)
    assert lesson is not None
    assert lesson.component == HarnessComponent.GUARDRAILS


def test_propose_probe_composer_weights():
    state = HarnessState(operational_capital=1000.0)
    composer = ProposeProbeComposer(state)
    lesson = HarnessLesson(
        id="lesson_test",
        symbol="CDNS",
        component=HarnessComponent.AGENT_WEIGHTS,
        lesson_text="Lower trend_following",
        evidence="Lost in whipsaw",
    )
    proposals = composer.propose_updates([lesson])
    assert "trend_following" in proposals["weights"]
    new_state = composer.compose_state(proposals)
    assert new_state.dynamic_weights_adjustment.get("trend_following") == -0.02


def test_harness_engine_lifecycle(tmp_path: Path):
    mem_file = tmp_path / "test_harness_memory.json"
    engine = HarnessEngine(memory_file=mem_file)
    assert engine.state.operational_capital == 1000.0

    diag = engine.record_closed_trade(
        symbol="PYPL",
        side="BUY",
        entry_price=58.0,
        exit_price=60.0,
        quantity=2.0,
        pnl_usd=4.0,
        pnl_pct=3.45,
    )
    assert diag.pnl_usd == 4.0
    assert len(engine.state.diagnoses) == 1
    assert mem_file.exists()

    summary = engine.get_summary()
    assert summary["operational_capital"] == 1000.0
    assert summary["total_diagnoses"] == 1
