"""AI Quant Trading Platform — Harness² Engine (Recursive Agent Harnessing).

Based on Google Research Harness² (arXiv: 2610.14224):
Implements dual-recursion agent harnessing:
1. Task-Level Recursion: Propose-Probe-Compose for trade execution, risk guardrails, and strategy ensemble.
2. Harnessing-Level Recursion: Continuous cross-trade reflection, forensic failure diagnosis, and persistent improvement library.
"""

from services.harness_engine.models import (
    FailureCategory,
    HarnessComponent,
    HarnessDiagnosis,
    HarnessLesson,
    HarnessState,
)
from services.harness_engine.reflector import ExperienceReflector
from services.harness_engine.composer import ProposeProbeComposer
from services.harness_engine.harness import HarnessEngine, get_harness_engine

__all__ = [
    "FailureCategory",
    "HarnessComponent",
    "HarnessDiagnosis",
    "HarnessLesson",
    "HarnessState",
    "ExperienceReflector",
    "ProposeProbeComposer",
    "HarnessEngine",
    "get_harness_engine",
]
