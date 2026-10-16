"""Model Training & Probability Calibration Engine."""

import uuid
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    roc_auc_score,
)
from services.ml_pipeline.models import DatasetSplit, ModelMetadata, ModelStatus


class QuantModelTrainer:
    """Trains Gradient Boosting quant models with probability calibration and strict auditing."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_child_samples: int = 10,
        random_state: int = 42,
    ):
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "verbosity": -1,
        }

    def train_and_calibrate(
        self,
        split: DatasetSplit,
        model_name: str = "QuantAlphaLGBM",
        version: str = "1.0.0",
    ) -> Tuple[object, ModelMetadata]:
        """Fits LightGBM on training set, calibrates on validation set, and records metrics.

        Returns:
            Tuple[object, ModelMetadata]: (calibrated_model, metadata)
        """
        import pandas as pd
        X_train_df = pd.DataFrame(split.X_train, columns=split.feature_names)
        X_val_df = pd.DataFrame(split.X_val, columns=split.feature_names)

        # 1. Base LightGBM classifier
        base_model = lgb.LGBMClassifier(**self.hyperparameters)
        base_model.fit(X_train_df, split.y_train)

        # 2. Probability Calibration
        unique_val_classes = np.unique(split.y_val)
        if len(unique_val_classes) > 1 and len(split.y_val) >= 20:
            try:
                from sklearn.frozen import FrozenEstimator
                calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method="sigmoid")
            except Exception:
                calibrator = CalibratedClassifierCV(estimator=base_model, cv="prefit", method="sigmoid")
            calibrator.fit(X_val_df, split.y_val)
            model_to_use = calibrator
        else:
            model_to_use = base_model

        # 3. Evaluate on Validation Set
        val_probs = model_to_use.predict_proba(X_val_df)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)

        # Compute metrics safely
        if len(np.unique(split.y_val)) > 1:
            auc = float(roc_auc_score(split.y_val, val_probs))
            ll = float(log_loss(split.y_val, val_probs))
        else:
            auc = 0.50
            ll = 0.693

        brier = float(brier_score_loss(split.y_val, val_probs))
        acc = float(accuracy_score(split.y_val, val_preds))
        prec = float(precision_score(split.y_val, val_preds, zero_division=0))

        # 4. Feature Importances from base tree model
        importances: Dict[str, float] = {}
        if hasattr(base_model, "feature_importances_"):
            raw_imp = base_model.feature_importances_
            total_imp = float(np.sum(raw_imp)) if np.sum(raw_imp) > 0 else 1.0
            for name, imp in zip(split.feature_names, raw_imp):
                importances[name] = round(float(imp) / total_imp, 4)

        metadata = ModelMetadata(
            model_id=f"ml_{uuid.uuid4().hex[:8]}",
            model_name=model_name,
            version=version,
            status=ModelStatus.CANDIDATE,
            feature_names=split.feature_names,
            hyperparameters=self.hyperparameters,
            roc_auc=round(auc, 4),
            brier_score=round(brier, 4),
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            log_loss=round(ll, 4),
            feature_importances=importances,
        )

        return model_to_use, metadata
