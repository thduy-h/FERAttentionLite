"""Predictor adapter for FERAttentionLite (trained on Kaggle, FER-2013).

Architecture is imported from `src.models.fer_attention_lite` -- never
redefined here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import torch

from src.models.fer_attention_lite import FERAttentionLite
from src.predictors.base import BaseEmotionPredictor, PredictionResult

FER_LABELS: List[str] = [
    "Angry",     # 0
    "Disgust",   # 1
    "Fear",      # 2
    "Happy",     # 3
    "Sad",       # 4
    "Surprise",  # 5
    "Neutral",   # 6
]

INPUT_SIZE = 48
NUM_CLASSES = 7


def resolve_device(device: str = "cpu") -> torch.device:
    """Map 'auto' | 'cpu' | 'cuda' to a torch.device."""
    device = (device or "cpu").lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available -> falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device != "cpu":
        raise ValueError(f"Unsupported device '{device}'. Use auto/cpu/cuda.")
    return torch.device("cpu")


class FERAttentionLitePredictor(BaseEmotionPredictor):
    """FERAttentionLite: grayscale 48x48, normalized to [-1, 1], 7 classes."""

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "cpu",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = resolve_device(device)
        self.checkpoint_info: Dict[str, Any] = {}
        self.model = self._load_model()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_model(self) -> FERAttentionLite:
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.weights_path}\n"
                "Copy Kaggle best.pt to weights/best.pt and run:\n"
                "  python -m src.tools.convert_to_cpu "
                "--input weights/best.pt --output weights/best_cpu.pt"
            )

        # weights_only=False: the checkpoint is our own trusted Kaggle file and
        # may contain metadata (epoch, metrics, config) besides tensors.
        checkpoint = torch.load(
            self.weights_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
            self.checkpoint_info = {
                k: v for k, v in checkpoint.items()
                if k != "model_state" and isinstance(v, (int, float, str))
            }
        elif isinstance(checkpoint, dict) and checkpoint and all(
            torch.is_tensor(v) for v in checkpoint.values()
        ):
            state = checkpoint  # raw state_dict
        else:
            raise ValueError(
                f"Unsupported checkpoint format in {self.weights_path}. "
                "Expected dict with 'model_state' or a raw state_dict."
            )

        model = FERAttentionLite(num_classes=NUM_CLASSES)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint does not match FERAttentionLite architecture "
                f"(strict=True).\n{exc}"
            ) from exc

        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    # Interface
    # ------------------------------------------------------------------ #
    def model_name(self) -> str:
        return "FERAttentionLite"

    @property
    def labels(self) -> List[str]:
        return list(FER_LABELS)

    def preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        """BGR crop -> gray -> 48x48 -> /255 -> (x-0.5)/0.5 -> [1,1,48,48]."""
        if face_bgr.ndim == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_bgr  # already single channel
        gray = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        x = gray.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1,1,48,48]
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

    def predict_batch(self, faces_bgr: List[np.ndarray]) -> List[PredictionResult]:
        if not faces_bgr:
            return []
        x = torch.cat([self.preprocess(f) for f in faces_bgr], dim=0)
        with torch.inference_mode():
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()
        return [self._postprocess(p) for p in probs]
