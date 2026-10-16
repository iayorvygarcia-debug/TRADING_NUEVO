"""Model Governance, Registry & Concept Drift Detection."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog
from services.ml_pipeline.models import DriftReport, ModelMetadata, ModelStatus

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Manages model artifacts, versioning, promotion gates, and governance."""

    def __init__(self, min_auc_threshold: float = 0.52, max_brier_threshold: float = 0.25):
        self.min_auc = min_auc_threshold
        self.max_brier = max_brier_threshold
        self._models: Dict[str, object] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._active_model_id: Optional[str] = None

    def register_model(self, model: object, metadata: ModelMetadata) -> str:
        """Registers a newly trained model candidate."""
        self._models[metadata.model_id] = model
        self._metadata[metadata.model_id] = metadata
        logger.info(
            "model_registered",
            model_id=metadata.model_id,
            name=metadata.model_name,
            version=metadata.version,
            auc=metadata.roc_auc,
            brier=metadata.brier_score,
        )
        return metadata.model_id

    def promote_to_active(self, model_id: str) -> bool:
        """Promotes a candidate model to ACTIVE if it satisfies governance performance gates."""
        if model_id not in self._metadata:
            raise ValueError(f"Model {model_id} not found in registry")

        meta = self._metadata[model_id]

        # Governance Performance Gate
        if meta.roc_auc < self.min_auc or meta.brier_score > self.max_brier:
            meta.status = ModelStatus.REJECTED
            logger.warning(
                "model_promotion_rejected_by_governance",
                model_id=model_id,
                auc=meta.roc_auc,
                min_auc=self.min_auc,
                brier=meta.brier_score,
                max_brier=self.max_brier,
            )
            return False

        # Retire currently active model if present
        if self._active_model_id and self._active_model_id in self._metadata:
            self._metadata[self._active_model_id].status = ModelStatus.RETIRED

        meta.status = ModelStatus.ACTIVE
        self._active_model_id = model_id
        logger.info("model_promoted_to_active", model_id=model_id, version=meta.version)
        return True

    def get_active_model(self) -> Tuple[Optional[object], Optional[ModelMetadata]]:
        """Returns the current active production model and its metadata."""
        if not self._active_model_id or self._active_model_id not in self._models:
            return None, None
        return self._models[self._active_model_id], self._metadata[self._active_model_id]

    def list_models(self) -> List[ModelMetadata]:
        """Returns metadata for all registered models."""
        return list(self._metadata.values())


class DriftDetector:
    """Calculates Population Stability Index (PSI) to detect feature & prediction distribution drift."""

    @staticmethod
    def calculate_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        num_buckets: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """Computes Population Stability Index between baseline and live distributions.

        PSI < 0.10: Stable / No drift
        0.10 <= PSI < 0.25: Moderate shift / Alert
        PSI >= 0.25: Significant drift / Retrain triggered
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create quantiles from expected
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Count frequencies
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_pct = expected_counts / max(1, len(expected))
        actual_pct = actual_counts / max(1, len(actual))

        # Add epsilon to prevent div-by-zero or log(0)
        expected_pct = np.clip(expected_pct, epsilon, 1.0)
        actual_pct = np.clip(actual_pct, epsilon, 1.0)

        # PSI formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(round(psi_val, 4))

    def evaluate_drift(
        self,
        metric_name: str,
        baseline_values: np.ndarray,
        live_values: np.ndarray,
        drift_threshold: float = 0.25,
    ) -> DriftReport:
        """Evaluates drift and returns a structured report."""
        psi = self.calculate_psi(baseline_values, live_values)
        is_drifted = psi >= drift_threshold

        return DriftReport(
            metric_name=metric_name,
            psi_score=psi,
            is_drifted=is_drifted,
            details=(
                f"Significant drift detected (PSI={psi} >= {drift_threshold}). Retraining advised."
                if is_drifted
                else f"Distribution stable (PSI={psi})."
            ),
        )
