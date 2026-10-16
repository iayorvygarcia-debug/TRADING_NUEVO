"""ML Signal Predictor & Inference Service."""

from typing import Optional
import numpy as np
import structlog
from services.ml_pipeline.governance import ModelRegistry
from services.ml_pipeline.models import PredictionResult

logger = structlog.get_logger(__name__)


class MLSignalPredictor:
    """Provides low-latency inference and quantitative trade signal filtering using active ML models."""

    def __init__(self, registry: ModelRegistry, custom_model: Optional[object] = None):
        self.registry = registry
        self._custom_model = custom_model

    def predict(self, symbol: str, feature_vector: np.ndarray) -> Optional[PredictionResult]:
        """Runs inference on a single feature vector.

        Args:
            symbol: Ticker symbol.
            feature_vector: 1D NumPy float32 vector matching the model's feature names.

        Returns:
            Optional[PredictionResult]: Prediction with calibrated probability and confidence.
        """
        active_meta = None
        model = self._custom_model
        if model is None:
            active_model, meta = self.registry.get_active_model()
            model = active_model
            active_meta = meta

        if model is None:
            return None

        # Reshape to 2D for sklearn/lightgbm
        X = feature_vector.reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        if active_meta and active_meta.feature_names and len(active_meta.feature_names) == X.shape[1]:
            import pandas as pd
            X = pd.DataFrame(X, columns=active_meta.feature_names)

        try:
            probs = model.predict_proba(X)[0]
            # Probability of Class 1 (Up)
            p_up = float(probs[1]) if len(probs) > 1 else float(probs[0])
            pred_label = 1 if p_up >= 0.50 else 0
            confidence = round(abs(p_up - 0.50) * 2.0, 4)

            return PredictionResult(
                symbol=symbol,
                raw_probability=round(p_up, 4),
                calibrated_probability=round(p_up, 4),
                predicted_label=pred_label,
                confidence=confidence,
            )
        except Exception as exc:
            logger.error("ml_inference_error", symbol=symbol, error=str(exc))
            return None

    def should_veto_signal(
        self,
        symbol: str,
        signal_direction: str,
        feature_vector: np.ndarray,
        min_probability_hurdle: float = 0.52,
    ) -> bool:
        """Determines if the ML model vetoes a rule-based strategy signal.

        - If signal is BUY and P(Up) < (1.0 - min_probability_hurdle) [e.g. < 0.48], veto = True.
        - If signal is SELL and P(Up) > min_probability_hurdle [e.g. > 0.52], veto = True.
        """
        pred = self.predict(symbol, feature_vector)
        if pred is None:
            # If no model active, do not veto
            return False

        if signal_direction.upper() == "BUY":
            # Model predicts strong downward bias against the buy signal
            if pred.calibrated_probability < (1.0 - min_probability_hurdle):
                logger.info(
                    "ml_vetoed_buy_signal",
                    symbol=symbol,
                    p_up=pred.calibrated_probability,
                )
                return True
        elif signal_direction.upper() == "SELL":
            # Model predicts strong upward bias against the short signal
            if pred.calibrated_probability > min_probability_hurdle:
                logger.info(
                    "ml_vetoed_sell_signal",
                    symbol=symbol,
                    p_up=pred.calibrated_probability,
                )
                return True

        return False
