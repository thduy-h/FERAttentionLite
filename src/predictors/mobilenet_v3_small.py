"""Predictor adapter for MobileNetV3-Small transfer learning on FER-2013."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import torch

from src.models.mobilenet_v3_small import build_mobilenet_v3_small
from src.predictors.base import BaseEmotionPredictor, PredictionResult
from src.predictors.fer_attention_lite import resolve_device

FER_LABELS: List[str] = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

INPUT_SIZE = 224
NUM_CLASSES = 7
IMAGENET_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)
IMAGENET_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


class MobileNetV3SmallPredictor(BaseEmotionPredictor):
    """RGB 224x224 + ImageNet normalization + 7 FER classes."""

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "cpu",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = resolve_device(device)
        self.checkpoint_info: Dict[str, Any] = {}
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Transfer checkpoint not found: {self.weights_path}\n"
                "Copy the Kaggle M2 best.pt to weights/m2_best.pt "
                "or pass --weights explicitly."
            )

        checkpoint = torch.load(
            self.weights_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
            self.checkpoint_info = {
                k: v
                for k, v in checkpoint.items()
                if k != "model_state" and isinstance(v, (int, float, str))
            }
        elif isinstance(checkpoint, dict) and checkpoint and all(
            torch.is_tensor(v) for v in checkpoint.values()
        ):
            state = checkpoint
        else:
            raise ValueError(
                f"Unsupported checkpoint format in {self.weights_path}. "
                "Expected dict with 'model_state' or a raw state_dict."
            )

        model = build_mobilenet_v3_small(
            num_classes=NUM_CLASSES,
            pretrained=False,
        )

        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint does not match the expected MobileNetV3-Small "
                f"FER architecture (strict=True).\n{exc}"
            ) from exc

        model.to(self.device)
        model.eval()
        return model

    def model_name(self) -> str:
        return "MobileNetV3-Small"

    @property
    def labels(self) -> List[str]:
        return list(FER_LABELS)

    def preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        """BGR face -> RGB 224x224 -> ImageNet normalize -> [1,3,224,224]."""
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face crop.")

        if face_bgr.ndim == 2:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(
            rgb,
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        x = rgb.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))

        tensor = torch.from_numpy(
            np.ascontiguousarray(x)
        ).unsqueeze(0)

        return tensor.to(self.device)

    def _postprocess(self, probs: np.ndarray) -> PredictionResult:
        idx = int(np.argmax(probs))
        return PredictionResult(
            label=FER_LABELS[idx],
            confidence=float(probs[idx]),
            probabilities=[float(p) for p in probs],
            labels=self.labels,
        )

    def predict(self, face_bgr: np.ndarray) -> PredictionResult:
        x = self.preprocess(face_bgr)

        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        return self._postprocess(probs)

    def predict_batch(
        self,
        faces_bgr: List[np.ndarray],
    ) -> List[PredictionResult]:
        if not faces_bgr:
            return []

        x = torch.cat(
            [self.preprocess(face) for face in faces_bgr],
            dim=0,
        )

        with torch.inference_mode():
            probs = torch.softmax(
                self.model(x),
                dim=1,
            ).cpu().numpy()

        return [
            self._postprocess(prob)
            for prob in probs
        ]
