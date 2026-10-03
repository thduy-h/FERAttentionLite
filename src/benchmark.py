"""Benchmark one face through any registered predictor on the local machine."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.predictors.factory import AVAILABLE_MODELS, create_predictor


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark registered predictor on one BGR face crop.")
    parser.add_argument("--model", default="cnn", choices=AVAILABLE_MODELS)
    parser.add_argument("--weights", default="weights/best_cpu.pt")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be >= 0 and --iters must be >= 1")

    predictor = create_predictor(args.model, args.weights, args.device)
    face_bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    device = torch.device(getattr(predictor, "device", "cpu"))

    def predict() -> object:
        return predictor.predict(face_bgr)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(args.warmup):
        predict()
    synchronize()
    started = time.perf_counter()
    for _ in range(args.iters):
        predict()
    synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0 / args.iters

    model = getattr(predictor, "model", None)
    parameters = (
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
        if isinstance(model, torch.nn.Module)
        else "unknown"
    )
    weights = Path(args.weights)
    checkpoint = f"{weights.stat().st_size / (1024 * 1024):.2f} MB" if weights.is_file() else "unknown"
    print(f"Model: {predictor.model_name()}")
    print(f"Device: {device.type.upper()}")
    print(f"Parameters: {parameters}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Latency: {latency_ms:.2f} ms/face (predictor including preprocessing)")
    print(f"Predictor FPS: {1000.0 / latency_ms:.1f}")
    print("Predictor FPS != full pipeline FPS (capture, detection, drawing and display).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
