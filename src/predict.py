"""Submission-compatible prediction helpers.

Model-specific preprocessing remains inside src/predictors/ so the realtime
application can switch between M1 and M2 without duplicating logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from src.predictors.base import PredictionResult
from src.predictors.factory import AVAILABLE_MODELS, create_predictor


def predict_face(
    face_bgr: np.ndarray,
    model_name: str,
    weights_path: Union[str, Path],
    device: str = "cpu",
) -> PredictionResult:
    """Load one registered predictor and classify one BGR face crop."""
    predictor = create_predictor(
        model_name=model_name,
        weights_path=weights_path,
        device=device,
    )
    return predictor.predict(face_bgr)


__all__ = [
    "AVAILABLE_MODELS",
    "PredictionResult",
    "create_predictor",
    "predict_face",
]
