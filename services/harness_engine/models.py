"""Data models for the Harness² Recursive Agent Harnessing Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """Forensic categories for trade losses and execution failures."""
    SIZING_OVEREXPOSURE = "SIZING_OVEREXPOSURE"          # Position was too large for capital base
    COUNTER_TREND_WHIPSAW = "COUNTER_TREND_WHIPSAW"      # Entered opposite to higher-timeframe trend
    SPREAD_SLIPPAGE_FRICTION = "SPREAD_SLIPPAGE_FRICTION"# Wide spread or execution slippage destroyed edge
    NEWS_SENTIMENT_TRAP = "NEWS_SENTIMENT_TRAP"          # Sell-the-news or fakeout despite positive headlines
    OVEREXTENDED_OSCILLATOR = "OVEREXTENDED_OSCILLATOR"  # Bought overbought (RSI>75) or shorted oversold
    SECTOR_CORRELATION = "SECTOR_CORRELATION"            # Clustered risk in correlated tech assets
    PREMATURE_STOP = "PREMATURE_STOP"                    # Stop-loss was too tight compared to normal asset ATR
    REGIME_MISALIGNMENT = "REGIME_MISALIGNMENT"          # Trend strategy executed in choppy/lateral market
    UNKNOWN = "UNKNOWN"


class HarnessComponent(str, Enum):
    """The editable components of an agent harness (Google Research Harness² taxonomy)."""
    SYSTEM_PROMPT = "system_prompt"       # LLM instructions & context (Gemini Analyzer)
    GUARDRAILS = "guardrails"             # Pre-trade checkers & dynamic spread thresholds
    AGENT_WEIGHTS = "agent_weights"       # Signal combiner committee weights
    POSITION_SIZING = "position_sizing"   # Sizing based on real operational capital
    TOOL_GUIDANCE = "tool_guidance"       # Market data & order parameters


class HarnessDiagnosis(BaseModel):
    """Forensic diagnostic report for a specific trade or trading day."""
    id: str
    symbol: str
    side: str                             # BUY / SELL (LONG / SHORT)
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    holding_seconds: float = 0.0
    failure_category: FailureCategory
    root_cause_explanation: str           # Detailed human-readable explanation
    mitigation_action: str                # What Harness² changed to prevent recurrence
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategies_involved: List[str] = Field(default_factory=list)
    news_score_at_entry: float = 0.0
    spread_at_entry: float = 0.0
    natr_at_entry: float = 0.0


class HarnessLesson(BaseModel):
    """Persistent, evidence-backed rule learned through trade reflection."""
    id: str
    symbol: Optional[str] = None          # Specific ticker or None for macro/general
    component: HarnessComponent
    lesson_text: str                      # Concise actionable guidance
    evidence: str                         # Historical trades or metrics supporting this
    confidence: float = 0.8
    times_applied: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


class HarnessState(BaseModel):
    """Active runtime harness configuration evolved by the engine."""
    operational_capital: float = 1_000.0  # Base capital ($1,000 USD)
    max_position_size_pct: float = 0.20   # Max 20% ($200) per position
    max_positions: int = 5
    dynamic_weights_adjustment: Dict[str, float] = Field(default_factory=dict)
    prompt_negative_examples: List[str] = Field(default_factory=list)
    tightened_spread_tickers: Dict[str, float] = Field(default_factory=dict)
    last_reflection_time: Optional[datetime] = None
    diagnoses: List[HarnessDiagnosis] = Field(default_factory=list)
    lessons: List[HarnessLesson] = Field(default_factory=list)
