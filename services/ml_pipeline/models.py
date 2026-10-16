"""Domain models for Machine Learning Pipeline & Model Governance."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    """Lifecycle state of a quantitative ML model."""
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class ModelMetadata(BaseModel):
    """Governance metadata for trained models."""
    model_id: str
    model_name: str
    version: str
    status: ModelStatus = ModelStatus.CANDIDATE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_names: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    
    # Validation Performance Metrics
    roc_auc: float = 0.50
    brier_score: float = 0.25
    accuracy: float = 0.50
    precision: float = 0.50
    log_loss: float = 0.693
    feature_importances: Dict[str, float] = Field(default_factory=dict)


class PredictionResult(BaseModel):
    """Output from ML model inference."""
    symbol: str
    raw_probability: float = Field(..., ge=0.0, le=1.0, description="Raw model probability P(Up)")
    calibrated_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated empirical probability")
    predicted_label: int = Field(..., description="1 for Up, 0 for Down/Flat")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction certainty |p - 0.5| * 2")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DriftReport(BaseModel):
    """Report on feature and prediction distribution stability."""
    metric_name: str
    psi_score: float = Field(..., description="Population Stability Index")
    is_drifted: bool = Field(..., description="True if PSI exceeds threshold (typically > 0.25)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[str] = None


class DatasetSplit(BaseModel):
    """Container for chronologically partitioned training/validation/testing matrices."""
    model_config = {"arbitrary_types_allowed": True}

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
