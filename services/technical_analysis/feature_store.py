"""
AI Quant Trading Platform — Feature Store.

In-memory and cache layer for computed quantitative features.
Enables:
1. Low-latency retrieval of the latest feature vectors by Strategy and Risk engines.
2. Fast universe scanning (e.g., retrieve latest features for all S&P 500 symbols).
3. Rolling buffer per symbol for sequence-based models or rolling feature normalization.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from core.logging import get_logger
from services.technical_analysis.feature_engine import FeatureVector

logger = get_logger("feature_store")


class FeatureStore:
    """
    In-memory rolling feature store.
    
    Keys format: `{symbol}:{timeframe}` (e.g. 'AAPL:1d', 'MSFT:15m')
    """

    def __init__(self, max_history_per_key: int = 200) -> None:
        self._max_history = max_history_per_key
        # Map: key -> deque of FeatureVector
        self._store: dict[str, deque[FeatureVector]] = defaultdict(
            lambda: deque(maxlen=self._max_history)
        )

    def _make_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}:{timeframe.lower()}"

    def save(self, feature_vector: FeatureVector) -> None:
        """Store a computed feature vector."""
        key = self._make_key(feature_vector.symbol, feature_vector.timeframe)
        self._store[key].append(feature_vector)
        logger.debug(
            "feature_saved",
            symbol=feature_vector.symbol,
            timeframe=feature_vector.timeframe,
            close=feature_vector.close_price,
            rsi=feature_vector.rsi_14,
        )

    def get(self, symbol: str, timeframe: str = "1d") -> FeatureVector | None:
        """Alias for get_latest."""
        return self.get_latest(symbol, timeframe)

    def get_latest(self, symbol: str, timeframe: str) -> FeatureVector | None:
        """Get the most recent FeatureVector for a symbol and timeframe."""
        key = self._make_key(symbol, timeframe)
        buffer = self._store.get(key)
        if buffer and len(buffer) > 0:
            return buffer[-1]
        return None

    def get_history(self, symbol: str, timeframe: str, count: int = 50) -> list[FeatureVector]:
        """Get recent sequence of FeatureVectors (oldest to newest)."""
        key = self._make_key(symbol, timeframe)
        buffer = self._store.get(key)
        if not buffer:
            return []
        items = list(buffer)
        return items[-count:]

    def get_universe_snapshot(
        self,
        symbols: list[str],
        timeframe: str,
    ) -> dict[str, FeatureVector]:
        """
        Batch retrieve latest features for a list of symbols.
        Used by the Universe Scanner / Opportunity Engine.
        """
        snapshot: dict[str, FeatureVector] = {}
        for sym in symbols:
            fv = self.get_latest(sym, timeframe)
            if fv is not None:
                snapshot[sym.upper()] = fv
        return snapshot

    def to_dataframe(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Convert recent stored features for a symbol to a Pandas DataFrame."""
        history = self.get_history(symbol, timeframe, count=count)
        if not history:
            return pd.DataFrame()
        records = [fv.to_dict() for fv in history]
        df = pd.DataFrame.from_records(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def clear(self) -> None:
        """Clear all stored features."""
        self._store.clear()


# Global in-memory singleton instance
_feature_store_instance: FeatureStore | None = None


def get_feature_store() -> FeatureStore:
    """Get or create singleton FeatureStore instance."""
    global _feature_store_instance
    if _feature_store_instance is None:
        _feature_store_instance = FeatureStore()
    return _feature_store_instance
