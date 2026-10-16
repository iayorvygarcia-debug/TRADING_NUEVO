"""Dataset Builder with Strict Anti-Lookahead Temporal Splitting."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from services.ml_pipeline.models import DatasetSplit
from services.technical_analysis.feature_engine import FeatureEngine


class DatasetBuilder:
    """Constructs machine learning datasets with strict chronological partitioning and embargoes."""

    def __init__(
        self,
        forward_horizon: int = 5,
        hurdle_rate: float = 0.003,
        embargo_bars: int = 5,
    ):
        self.forward_horizon = forward_horizon
        self.hurdle_rate = hurdle_rate
        self.embargo_bars = embargo_bars
        self.feature_engine = FeatureEngine()

    def build_features_and_labels(
        self, df: pd.DataFrame, symbol: str = "ASSET"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extracts stationary features and forward directional labels from OHLCV bars.

        Args:
            df: DataFrame containing open, high, low, close, volume.
            symbol: Ticker symbol.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: (X feature matrix, y labels array, feature names)
        """
        if len(df) < 50:
            raise ValueError(f"Insufficient bars to build feature dataset: got {len(df)}, need >= 50")

        # Sort chronologically
        df = df.sort_index().copy()

        # Compute forward returns: (close[t + H] - close[t]) / close[t]
        close = df["close"].values
        forward_returns = np.full(len(close), np.nan)
        forward_returns[:-self.forward_horizon] = (
            close[self.forward_horizon:] - close[:-self.forward_horizon]
        ) / close[:-self.forward_horizon]

        # Target label: 1 if return >= hurdle_rate, else 0
        labels = (forward_returns >= self.hurdle_rate).astype(np.int32)

        feature_matrix_list: List[np.ndarray] = []
        feature_names: List[str] = []

        # Generate feature vectors bar-by-bar to ensure strict anti-lookahead
        # (Start from index 35 where indicators have stabilized)
        start_idx = 35
        end_idx = len(df) - self.forward_horizon

        for i in range(start_idx, end_idx):
            window = df.iloc[: i + 1]
            f_vec = self.feature_engine.compute_features(df=window, symbol=symbol)
            if not feature_names:
                feature_names = f_vec.feature_names()
            feature_matrix_list.append(f_vec.to_ml_features())

        X = np.array(feature_matrix_list, dtype=np.float32)
        y = labels[start_idx:end_idx]

        # Handle any NaN/inf in X
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        return X, y, feature_names

    def split_temporal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> DatasetSplit:
        """Splits chronologically into Train, Validation, and Test sets with purge embargoes.

        Strictly NO random shuffling to prevent temporal data leakage.
        """
        n = len(X)
        if n < 30:
            raise ValueError(f"Dataset too small to partition: {n} samples")

        train_end = int(n * train_ratio)
        val_start = train_end + self.embargo_bars

        val_end = int(n * (train_ratio + val_ratio))
        test_start = val_end + self.embargo_bars

        if val_start >= val_end or test_start >= n:
            # Fallback for smaller sample sets: minimal embargo
            val_start = train_end
            val_end = int(n * (train_ratio + val_ratio))
            test_start = val_end

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]
        X_test, y_test = X[test_start:], y[test_start:]

        return DatasetSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
        )
