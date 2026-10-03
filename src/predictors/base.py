"""Common interface for every emotion predictor.

The app only talks to `BaseEmotionPredictor` and `PredictionResult`.
Each concrete predictor owns its architecture, checkpoint loading,
preprocessing, label order and inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probabilities: List[float]
    # Labels in the same order as `probabilities` (model specific).
    labels: Optional[List[str]] = None


class BaseEmotionPredictor(ABC):
    """Abstract emotion predictor.

    Subclasses must implement `predict` and `model_name`.
    """

    @abstractmethod
    def predict(self, face_bgr: np.ndarray) -> PredictionResult:
        """Predict emotion for a single BGR face crop (H, W, 3, uint8)."""

    @abstractmethod
    def model_name(self) -> str:
        """Human readable model name (shown in the UI overlay)."""

    @property
    def labels(self) -> List[str]:
        """Class labels in model output order. Override in subclasses."""
        return []

    def predict_batch(self, faces_bgr: List[np.ndarray]) -> List[PredictionResult]:
        """Default batch implementation: loop over `predict`.

        Subclasses may override for real batched inference.
        """
        return [self.predict(face) for face in faces_bgr]
