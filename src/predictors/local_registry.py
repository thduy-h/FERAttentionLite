"""Private registrations for models kept only in this local workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from src.predictors.base import BaseEmotionPredictor

LOCAL_MODELS = ("cnn",)


def create_local_predictor(
    model_name: str,
    weights_path: Union[str, Path],
    device: str = "cpu",
) -> BaseEmotionPredictor:
    if model_name == "cnn":
        from src.predictors.fer_attention_lite import FERAttentionLitePredictor

        return FERAttentionLitePredictor(weights_path=weights_path, device=device)
    raise ValueError(f"Unknown local model: {model_name}")
