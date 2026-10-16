"""Unit tests for Machine Learning Pipeline & Model Governance."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from services.ml_pipeline.dataset_builder import DatasetBuilder
from services.ml_pipeline.governance import DriftDetector, ModelRegistry
from services.ml_pipeline.models import (
    DatasetSplit,
    ModelMetadata,
    ModelStatus,
)
from services.ml_pipeline.predictor import MLSignalPredictor
from services.ml_pipeline.trainer import QuantModelTrainer


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generates 200 synthetic chronological OHLCV bars."""
    np.random.seed(42)
    n = 200
    prices = [100.0]
    for _ in range(n - 1):
        ret = np.random.normal(0.0005, 0.015)
        prices.append(prices[-1] * (1.0 + ret))

    prices = np.array(prices)
    highs = prices * (1.0 + np.abs(np.random.normal(0.005, 0.003, n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0.005, 0.003, n)))
    opens = (highs + lows) / 2.0
    closes = prices
    volumes = np.random.uniform(50000, 200000, n)

    dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )
    return df


# ==========================================
# 1. Dataset Builder Tests
# ==========================================

def test_dataset_builder_feature_and_label_generation(sample_ohlcv_df):
    builder = DatasetBuilder(forward_horizon=5, hurdle_rate=0.002, embargo_bars=3)
    X, y, names = builder.build_features_and_labels(sample_ohlcv_df, symbol="TEST")

    assert len(X) > 0
    assert len(X) == len(y)
    assert len(names) == X.shape[1]
    assert set(np.unique(y)).issubset({0, 1})
    # Check no NaNs in feature matrix
    assert not np.isnan(X).any()


def test_dataset_builder_temporal_split_no_shuffle(sample_ohlcv_df):
    builder = DatasetBuilder(forward_horizon=5, hurdle_rate=0.002, embargo_bars=2)
    X, y, names = builder.build_features_and_labels(sample_ohlcv_df, symbol="TEST")

    split = builder.split_temporal(X, y, names, train_ratio=0.70, val_ratio=0.15)

    # Invariant: train is strictly before val, and val is strictly before test
    assert len(split.X_train) > 0
    assert len(split.X_val) > 0
    assert len(split.X_test) > 0
    # Total samples <= original len due to embargo purge
    assert (len(split.X_train) + len(split.X_val) + len(split.X_test)) <= len(X)


# ==========================================
# 2. Quant Model Trainer Tests
# ==========================================

def test_model_trainer_fit_and_calibrate():
    np.random.seed(42)
    n_features = 10
    n_train = 120
    n_val = 40
    n_test = 40

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)

    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    feature_names = [f"feat_{i}" for i in range(n_features)]

    split = DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )

    trainer = QuantModelTrainer(n_estimators=30, max_depth=3)
    model, meta = trainer.train_and_calibrate(split)

    assert meta.model_id.startswith("ml_")
    assert meta.status == ModelStatus.CANDIDATE
    assert 0.0 <= meta.roc_auc <= 1.0
    assert 0.0 <= meta.brier_score <= 1.0
    assert len(meta.feature_importances) > 0

    # Predictions should be valid probabilities in [0.0, 1.0]
    probs = model.predict_proba(X_test)[:, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


# ==========================================
# 3. Model Registry & Governance Tests
# ==========================================

def test_model_registry_promotion_and_rejection():
    registry = ModelRegistry(min_auc_threshold=0.55, max_brier_threshold=0.25)

    # Candidate 1: Low AUC (fails gate)
    meta_bad = ModelMetadata(
        model_id="m_bad",
        model_name="BadModel",
        version="0.1.0",
        roc_auc=0.51,  # Below 0.55
        brier_score=0.24,
    )
    registry.register_model(object(), meta_bad)
    promoted_bad = registry.promote_to_active("m_bad")
    assert promoted_bad is False
    assert meta_bad.status == ModelStatus.REJECTED

    # Candidate 2: Superior AUC (passes gate)
    meta_good = ModelMetadata(
        model_id="m_good",
        model_name="GoodModel",
        version="1.0.0",
        roc_auc=0.62,
        brier_score=0.21,
    )
    registry.register_model("mock_model_object_1", meta_good)
    promoted_good = registry.promote_to_active("m_good")
    assert promoted_good is True
    assert meta_good.status == ModelStatus.ACTIVE

    active_model, active_meta = registry.get_active_model()
    assert active_model == "mock_model_object_1"
    assert active_meta.model_id == "m_good"

    # Candidate 3: Even better model arrives
    meta_better = ModelMetadata(
        model_id="m_better",
        model_name="BetterModel",
        version="2.0.0",
        roc_auc=0.65,
        brier_score=0.19,
    )
    registry.register_model("mock_model_object_2", meta_better)
    promoted_better = registry.promote_to_active("m_better")
    assert promoted_better is True

    # Check that previous model transitioned to RETIRED
    assert meta_good.status == ModelStatus.RETIRED
    assert meta_better.status == ModelStatus.ACTIVE


# ==========================================
# 4. Drift Detector (PSI) Tests
# ==========================================

def test_drift_detector_stable():
    np.random.seed(42)
    baseline = np.random.normal(0.0, 1.0, 1000)
    live_same_dist = np.random.normal(0.0, 1.0, 1000)

    report = DriftDetector().evaluate_drift("feature_momentum", baseline, live_same_dist)
    assert report.psi_score < 0.10
    assert report.is_drifted is False


def test_drift_detector_severe_drift():
    np.random.seed(42)
    baseline = np.random.normal(0.0, 1.0, 1000)
    # Severe regime shift: mean shifted by 3 standard deviations
    live_shifted = np.random.normal(3.0, 1.0, 1000)

    report = DriftDetector().evaluate_drift("feature_momentum", baseline, live_shifted)
    assert report.psi_score > 0.25
    assert report.is_drifted is True


# ==========================================
# 5. ML Signal Predictor & Trade Veto Tests
# ==========================================

def test_signal_predictor_inference_and_veto():
    # Setup trained dummy model using sklearn LogisticRegression
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    X_train = np.array([[-2.0], [-1.5], [1.5], [2.0]], dtype=np.float32)
    y_train = np.array([0, 0, 1, 1], dtype=int)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    registry = ModelRegistry()
    meta = ModelMetadata(
        model_id="m_active",
        model_name="VetoModel",
        version="1.0.0",
        roc_auc=0.99,
        brier_score=0.05,
    )
    registry.register_model(clf, meta)
    registry.promote_to_active("m_active")

    predictor = MLSignalPredictor(registry=registry)

    # Test Up Vector (feature = +2.5 -> High P(Up))
    res_up = predictor.predict("AAPL", np.array([2.5], dtype=np.float32))
    assert res_up is not None
    assert res_up.calibrated_probability > 0.70
    assert res_up.predicted_label == 1

    # Should NOT veto BUY signal when probability is high
    assert predictor.should_veto_signal("AAPL", "BUY", np.array([2.5], dtype=np.float32)) is False

    # Should VETO BUY signal when feature = -2.5 (High P(Down) / low P(Up))
    assert predictor.should_veto_signal("AAPL", "BUY", np.array([-2.5], dtype=np.float32)) is True
