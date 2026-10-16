"""
AI Quant Trading Platform — Universe Scanner & Opportunity Ranker.

Orchestrates universe scanning, screening, composite scoring, and top-tier ranking.
Provides:
- Batch async scanning across hundreds or thousands of tickers
- Automatic sorting by multi-factor OpportunityScore
- Top-N selection ready for Strategy Engine signal dispatch
- Formatted textual scanning reports
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from core.logging import get_logger
from services.opportunity_engine.screener import OpportunityScreener, ScreeningResult
from services.opportunity_engine.scorer import OpportunityScorer, ScoredOpportunity
from services.opportunity_engine.universe import UniverseManager
from services.technical_analysis.feature_engine import FeatureEngine, FeatureVector
from services.technical_analysis.feature_store import FeatureStore, get_feature_store

logger = get_logger("universe_scanner")


class UniverseScanner:
    """
    Automated Universe Scanner and Opportunity Ranking Engine.
    """

    def __init__(
        self,
        screener: OpportunityScreener | None = None,
        feature_store: FeatureStore | None = None,
        max_concurrency: int = 20,
    ) -> None:
        self.screener = screener or OpportunityScreener()
        self.feature_store = feature_store or get_feature_store()
        self.max_concurrency = max_concurrency
        self._batch_offset: int = 0

    def scan_from_features(
        self,
        features_list: list[FeatureVector],
        min_score: float = 50.0,
        top_n: int = 10,
    ) -> list[ScoredOpportunity]:
        """
        Scan and rank a pre-computed list of FeatureVectors.
        """
        passed_scores: list[ScoredOpportunity] = []

        for fv in features_list:
            screen_res = self.screener.screen(fv)
            if screen_res.passed:
                scored = OpportunityScorer.score(fv)
                if scored.composite_score >= min_score:
                    passed_scores.append(scored)

        # Sort descending by composite score
        passed_scores.sort(key=lambda x: x.composite_score, reverse=True)
        return passed_scores[:top_n]

    async def scan_universe_data(
        self,
        data_map: dict[str, pd.DataFrame],
        timeframe: str = "1d",
        min_score: float = 50.0,
        top_n: int = 10,
    ) -> list[ScoredOpportunity]:
        """
        Scan a mapping of `{symbol: DataFrame}` across an entire universe.
        Uses semaphore to manage parallel computation cleanly.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        opportunities: list[ScoredOpportunity] = []

        async def _process_symbol(sym: str, df: pd.DataFrame) -> None:
            async with semaphore:
                try:
                    if len(df) < FeatureEngine.MIN_BARS_REQUIRED:
                        return

                    # Compute features
                    fv = FeatureEngine.compute_features(df, symbol=sym, timeframe=timeframe)
                    # Cache in FeatureStore
                    self.feature_store.save(fv)

                    # Screen
                    screen_res = self.screener.screen(fv)
                    if screen_res.passed:
                        scored = OpportunityScorer.score(fv)
                        if scored.composite_score >= min_score:
                            opportunities.append(scored)
                except Exception as e:
                    logger.debug("scan_symbol_error", symbol=sym, error=str(e))

        tasks = [_process_symbol(sym, df) for sym, df in data_map.items()]
        await asyncio.gather(*tasks)

        # Sort descending by composite score
        opportunities.sort(key=lambda x: x.composite_score, reverse=True)
        logger.info(
            "universe_scan_completed",
            total_scanned=len(data_map),
            passed_count=len(opportunities),
            top_symbol=opportunities[0].symbol if opportunities else None,
        )
        return opportunities[:top_n]

    async def scan_universe(
        self,
        symbols: list[str],
        top_n: int = 10,
        min_score: float = 50.0,
    ) -> list[ScoredOpportunity]:
        """
        Scans symbols dynamically rotating across the universe using Twelve Data.
        Each scan analyzes the next batch of tickers and re-ranks all discovered setups.
        """
        if not symbols:
            return []

        # Rotate batch of symbols to evaluate on each cycle (5 symbols per scan)
        batch_size = min(len(symbols), 5)
        start_idx = self._batch_offset % len(symbols)
        target_syms = [symbols[(start_idx + i) % len(symbols)] for i in range(batch_size)]
        self._batch_offset = (self._batch_offset + batch_size) % len(symbols)

        try:
            from services.market_data.twelve_data_adapter import TwelveDataAdapter
            adapter = TwelveDataAdapter()
            await adapter.connect()
            for sym in target_syms:
                try:
                    raw_bars = await adapter.get_time_series(sym, interval="1day", outputsize=60)
                    if raw_bars and len(raw_bars) >= FeatureEngine.MIN_BARS_REQUIRED:
                        df = pd.DataFrame(raw_bars).copy()
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        df = df.sort_values("datetime")
                        for col in ["open", "high", "low", "close", "volume"]:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                        df = df.set_index("datetime")
                        fv = FeatureEngine.compute_features(df, symbol=sym, timeframe="1d")
                        self.feature_store.save(fv)
                except Exception as err:
                    logger.debug("scan_symbol_fetch_error", symbol=sym, error=str(err))
            await adapter.disconnect()
        except Exception as exc:
            logger.debug("auto_scan_twelve_data_error", error=str(exc))

        # Collect all active FeatureVectors across the symbols universe
        features_list: list[FeatureVector] = []
        for sym in symbols:
            fv = self.feature_store.get(sym)
            if fv is not None:
                features_list.append(fv)

        return self.scan_from_features(features_list, min_score=min_score, top_n=top_n)


    @classmethod
    def generate_report(cls, opportunities: list[ScoredOpportunity]) -> str:
        """Format scanning results as an executive text table."""
        if not opportunities:
            return "No opportunities passed screening criteria (NO TRADES)."

        lines = [
            "================================================================================",
            "                   TOP QUANTITATIVE OPPORTUNITIES RANKING                       ",
            "================================================================================",
            f"{'Rank':<5} {'Symbol':<8} {'Score':<7} {'Bias':<9} {'Setup Type':<26} {'Key Reasons'}",
            "--------------------------------------------------------------------------------",
        ]

        for i, opp in enumerate(opportunities, 1):
            reasons_summary = ", ".join(opp.reasons[:2]) if opp.reasons else "Clean setup"
            lines.append(
                f"{i:<5} {opp.symbol:<8} {opp.composite_score:<7.1f} {opp.direction.upper():<9} "
                f"{opp.setup_type:<26} {reasons_summary}"
            )

        lines.append("================================================================================")
        return "\n".join(lines)
