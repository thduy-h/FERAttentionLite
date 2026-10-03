"""Register model-specific predictors here without changing the app."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from src.predictors.base import BaseEmotionPredictor

AVAILABLE_MODELS = ("cnn", "transfer", "friend")


def create_predictor(
    model_name: str,
    weights_path: Union[str, Path],
    device: str = "cpu",
) -> BaseEmotionPredictor:
    name = (model_name or "").lower().strip()

    if name == "cnn":
        from src.predictors.fer_attention_lite import FERAttentionLitePredictor

        return FERAttentionLitePredictor(
            weights_path=weights_path,
            device=device,
        )

    if name == "transfer":
        from src.predictors.mobilenet_v3_small import MobileNetV3SmallPredictor

        return MobileNetV3SmallPredictor(
            weights_path=weights_path,
            device=device,
        )

    if name == "friend":
        raise NotImplementedError(
            "Add FriendModelPredictor in src/predictors/friend_model.py "
            "and register it in src/predictors/factory.py."
        )

    raise ValueError(
        f"Unknown model '{model_name}'. Available: {', '.join(AVAILABLE_MODELS)}"
    )
