"""Facial Emotion Recognition -- model-agnostic local inference app.

Supports image, video file, webcam index and IP camera (HTTP/RTSP) sources.

The app calls `predictor.predict(face)` and reads
`result.label / confidence / probabilities`.
All model-specific logic (architecture, preprocessing, labels) lives in
`src/predictors/`.

Examples:
    python -m src.app --source test.jpg --model cnn --weights weights/best_cpu.pt
    python -m src.app --source test.jpg --model transfer --weights weights/m2_best.pt
    python -m src.app --source 0 --model cnn
    python -m src.app --source http://192.168.1.15:8080/video --model cnn
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.detector import BBox, FaceDetector, crop_face
from src.predictors.base import BaseEmotionPredictor, PredictionResult
from src.predictors.factory import AVAILABLE_MODELS, create_predictor
from src.utils.drawing import draw_prediction, draw_status, resize_for_display

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WINDOW_NAME = "Facial Emotion Recognition"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
QUIT_KEYS = {ord("q"), ord("Q"), 27}  # q / ESC
FPS_EMA_ALPHA = 0.1

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "active": "cnn",
        "cnn_weights": "weights/best_cpu.pt",
        "transfer_weights": "weights/m2_best.pt",
        "friend_weights": "weights/friend.pt",
        "confidence_threshold": 0.40,
    },
    "camera": {
        "source": "",
        "reconnect_delay": 1.0,
    },
    "detector": {
        "scale_factor": 1.1,
        "min_neighbors": 5,
        "min_face_size": 40,
        "margin": 0.10,
        "detect_scale": 1.0,
    },
    "runtime": {
        "device": "cpu",
        "display_width": 960,
        "skip_frames": 0,
    },
}

Detection = Tuple[BBox, PredictionResult]


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Load YAML config merged over defaults. Missing file -> defaults."""
    cfg_path = Path(path) if path else (
        LOCAL_CONFIG_PATH if LOCAL_CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH
    )
    if not cfg_path.exists():
        if path:
            print(f"[WARN] Config not found: {cfg_path} -> using defaults.")
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # malformed YAML etc.
        print(f"[WARN] Failed to read config {cfg_path}: {exc} -> using defaults.")
        return copy.deepcopy(DEFAULT_CONFIG)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {cfg_path}")
    return _deep_merge(DEFAULT_CONFIG, data)


@dataclass
class Settings:
    source: str
    model: str
    weights: str
    device: str
    confidence: float
    display_width: int
    detector_scale: float
    min_face: int
    scale_factor: float
    min_neighbors: int
    margin: float
    skip_frames: int
    reconnect_delay: float
    save_output: Optional[str]
    display: bool


def _resolve_path(p: str) -> str:
    """Relative paths are resolved against CWD first, then project root."""
    path = Path(p)
    if path.is_absolute() or path.exists():
        return str(path)
    candidate = PROJECT_ROOT / path
    return str(candidate) if candidate.exists() else str(path)


def build_settings(args: argparse.Namespace) -> Settings:
    """CLI arguments override config.yaml, which overrides built-in defaults."""
    cfg = load_config(args.config)
    m, cam, det, rt = cfg["model"], cfg["camera"], cfg["detector"], cfg["runtime"]

    def pick(cli_value: Any, cfg_value: Any) -> Any:
        return cli_value if cli_value is not None else cfg_value

    model = str(pick(args.model, m.get("active", "cnn"))).lower()

    if args.weights is not None:
        weights = _resolve_path(args.weights)
    else:
        weights = _resolve_path(str(m.get(f"{model}_weights", f"weights/{model}.pt")))

    source = pick(args.source, cam.get("source") or None)
    if not source and source != 0:
        raise SystemExit("[ERROR] No --source given and camera.source empty in config.")

    return Settings(
        source=str(source),
        model=model,
        weights=weights,
        device=str(pick(args.device, rt.get("device", "cpu"))),
        confidence=float(pick(args.confidence, m.get("confidence_threshold", 0.40))),
        display_width=int(pick(args.display_width, rt.get("display_width", 960))),
        detector_scale=float(pick(args.detector_scale, det.get("detect_scale", 1.0))),
        min_face=int(pick(args.min_face, det.get("min_face_size", 40))),
        scale_factor=float(det.get("scale_factor", 1.1)),
        min_neighbors=int(det.get("min_neighbors", 5)),
        margin=float(det.get("margin", 0.10)),
        skip_frames=int(pick(args.skip_frames, rt.get("skip_frames", 0))),
        reconnect_delay=float(cam.get("reconnect_delay", 1.0)),
        save_output=args.save_output,
        display=not args.no_display,
    )


# ---------------------------------------------------------------------- #
# Source handling
# ---------------------------------------------------------------------- #
def parse_source(source: Union[str, int]) -> Tuple[str, Union[str, int]]:
    """Return (kind, value) where kind in {'ip', 'webcam', 'image', 'video'}."""
    s = str(source).strip()
    lower = s.lower()
    if lower.startswith(("http://", "https://", "rtsp://")):
        return "ip", s
    if s.isdigit():
        return "webcam", int(s)
    if lower == "droidcam":
        return "webcam", "droidcam"
    if s.startswith("/dev/video") and s[10:].isdigit():
        return "webcam", s
    if Path(lower).suffix in IMAGE_EXTS:
        return "image", s
    return "video", s


def find_droidcam_device() -> Optional[str]:
    """Find the current /dev/videoN whose V4L2 name is DroidCam."""
    for entry in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            if "droidcam" in (entry / "name").read_text().strip().lower():
                device = Path("/dev") / entry.name
                if device.exists():
                    return str(device)
        except OSError:
            continue
    return None


def open_capture(kind: str, value: Union[str, int]) -> cv2.VideoCapture:
    """Open a VideoCapture. For IP cameras use FFmpeg with timeouts if possible."""
    if kind == "webcam" and value == "droidcam":
        device = find_droidcam_device()
        if device is None:
            return cv2.VideoCapture()
        value = device
    cap: Optional[cv2.VideoCapture] = None
    if kind == "ip":
        open_to = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_to = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if open_to is not None and read_to is not None:
            try:
                cap = cv2.VideoCapture(value, cv2.CAP_FFMPEG, [open_to, 5000, read_to, 5000])
            except (cv2.error, TypeError):
                cap = None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            try:
                cap = cv2.VideoCapture(value)
            except cv2.error:
                cap = cv2.VideoCapture()
    else:
        try:
            cap = cv2.VideoCapture(value)
        except cv2.error:
            cap = cv2.VideoCapture()

    if cap.isOpened() and kind in ("ip", "webcam"):
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low (ignored if unsupported)
        except cv2.error:
            pass
    return cap


# ---------------------------------------------------------------------- #
# Core per-frame processing (model-agnostic)
# ---------------------------------------------------------------------- #
def process_frame(
    frame: np.ndarray,
    detector: FaceDetector,
    predictor: BaseEmotionPredictor,
) -> List[Detection]:
    boxes = detector.detect(frame)
    crops, kept = [], []
    for box in boxes:
        crop = crop_face(frame, box)
        if crop is not None:
            crops.append(crop)
            kept.append(box)
    # Both built-in predictors implement true batched inference. The base
    # class also provides a safe loop fallback for third-party predictors.
    results = predictor.predict_batch(crops)
    return list(zip(kept, results))


def annotate(
    frame: np.ndarray,
    detections: List[Detection],
    threshold: float,
) -> None:
    for box, res in detections:
        draw_prediction(frame, box, res.label, res.confidence, threshold)


class DisplayWindow:
    """cv2.imshow wrapper that degrades gracefully when no GUI is available."""

    def __init__(self, enabled: bool, display_width: int) -> None:
        self.enabled = enabled
        self.display_width = display_width

    def show(self, frame: np.ndarray, wait_ms: int = 1) -> int:
        """Show frame, return key code (or -1)."""
        if not self.enabled:
            return -1
        try:
            cv2.imshow(WINDOW_NAME, resize_for_display(frame, self.display_width))
            return cv2.waitKey(wait_ms) & 0xFF
        except cv2.error as exc:
            print(f"[WARN] Display unavailable ({exc.__class__.__name__}); "
                  "continuing without window. Use --no-display to silence.")
            self.enabled = False
            return -1

    def poll_key(self, wait_ms: int = 1) -> int:
        if not self.enabled:
            return -1
        try:
            return cv2.waitKey(wait_ms) & 0xFF
        except cv2.error:
            return -1

    def close(self) -> None:
        if self.enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# ---------------------------------------------------------------------- #
# Modes
# ---------------------------------------------------------------------- #
def run_image(
    path: str,
    settings: Settings,
    detector: FaceDetector,
    predictor: BaseEmotionPredictor,
) -> int:
    image = cv2.imread(path)
    if image is None:
        print(f"[ERROR] Cannot read image: {path}")
        return 1

    detections = process_frame(image, detector, predictor)
    if not detections:
        print("[INFO] No face detected. Saving original image with status overlay.")
    for i, (box, res) in enumerate(detections, 1):
        tag = "" if res.confidence >= settings.confidence else "  (uncertain)"
        print(f"  Face {i}: bbox={box}  {res.label} {res.confidence * 100:.1f}%{tag}")

    annotate(image, detections, settings.confidence)
    draw_status(image, None, len(detections), predictor.model_name())

    if settings.save_output:
        out_path = Path(settings.save_output)
    else:
        out_path = OUTPUT_DIR / f"{Path(path).stem}_result.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        print(f"[ERROR] Cannot save image: {out_path}")
        return 1
    print(f"[INFO] Saved: {out_path}")

    window = DisplayWindow(settings.display, settings.display_width)
    if window.enabled:
        print("[INFO] Press any key to close the window.")
        window.show(image, wait_ms=0)
    window.close()
    return 0


def run_stream(
    kind: str,
    value: Union[str, int],
    settings: Settings,
    detector: FaceDetector,
    predictor: BaseEmotionPredictor,
) -> int:
    """Video file, webcam or IP camera loop with reconnect for live sources."""
    is_live = kind in ("ip", "webcam")
    label = f"{kind}:{value}"

    cap = open_capture(kind, value)
    if not cap.isOpened():
        if not is_live:
            print(f"[ERROR] Cannot open video: {value}")
            return 1
        print(f"[WARN] Cannot open {label}. Will keep retrying (q/ESC or Ctrl+C to quit).")

    window = DisplayWindow(settings.display, settings.display_width)
    writer: Optional[cv2.VideoWriter] = None
    model_name = predictor.model_name()

    frame_idx = 0
    last_detections: List[Detection] = []
    fps_ema: Optional[float] = None
    print(f"[INFO] Running on {label} -- press q or ESC to quit.")
    try:
        while True:
            frame_start = time.perf_counter()
            try:
                ok, frame = (cap.read() if cap.isOpened() else (False, None))
            except cv2.error as exc:
                print(f"[WARN] Capture read failed for {label}: {exc}")
                ok, frame = False, None

            if not ok or frame is None:
                if not is_live:
                    print("[INFO] End of video.")
                    break
                # ---- live source lost: release, wait, reconnect ----
                print(f"[WARN] Lost connection to {label}. "
                      f"Reconnecting in {settings.reconnect_delay:.1f}s ...")
                cap.release()
                end = time.time() + settings.reconnect_delay
                while time.time() < end:
                    # Keep the window responsive so q/ESC still works.
                    if window.enabled:
                        if window.poll_key(50) in QUIT_KEYS:
                            return 0
                    else:
                        time.sleep(0.05)
                cap = open_capture(kind, value)
                if cap.isOpened():
                    print(f"[INFO] Reconnected to {label}.")
                last_detections = []
                continue

            # ---- detect + classify (every skip_frames+1 frames) ----
            if frame_idx % (settings.skip_frames + 1) == 0:
                last_detections = process_frame(frame, detector, predictor)
            frame_idx += 1

            annotate(frame, last_detections, settings.confidence)

            # Show the previous completed frame's full-pipeline FPS.
            draw_status(frame, fps_ema, len(last_detections), model_name)

            if settings.save_output:
                if writer is None:
                    writer = _create_writer(settings.save_output, cap, frame, is_live)
                if writer is not None:
                    writer.write(frame)

            key = window.show(frame, 1)
            elapsed = time.perf_counter() - frame_start
            if elapsed > 0:
                fps_now = 1.0 / elapsed
                fps_ema = fps_now if fps_ema is None else (
                    FPS_EMA_ALPHA * fps_now + (1 - FPS_EMA_ALPHA) * fps_ema
                )
            if key in QUIT_KEYS:
                break
            if not window.enabled and not settings.display and frame_idx % 100 == 0:
                print(f"[INFO] frames={frame_idx} fps={fps_ema or 0:.1f} "
                      f"faces={len(last_detections)}")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Saved video: {settings.save_output}")
        window.close()
    return 0


def _create_writer(
    path: str,
    cap: cv2.VideoCapture,
    frame: np.ndarray,
    is_live: bool,
) -> Optional[cv2.VideoWriter]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if is_live or not (1.0 <= fps <= 120.0):
        fps = 20.0
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"[WARN] Cannot open VideoWriter for {out}; output will not be saved.")
        return None
    print(f"[INFO] Recording to {out} ({w}x{h} @ {fps:.1f} FPS)")
    return writer


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.app",
        description="Facial Emotion Recognition: image / video / webcam / IP camera.",
    )
    p.add_argument("--source", default=None,
                   help="Image/video path, webcam index or /dev/videoN, droidcam, or http(s)/rtsp URL.")
    p.add_argument("--model", default=None, choices=AVAILABLE_MODELS,
                   help="Predictor name (default from config: model.active).")
    p.add_argument("--weights", default=None,
                   help="Checkpoint path (default from config: model.<model>_weights).")
    p.add_argument("--confidence", type=float, default=None,
                   help="Confidence threshold; below it the face is shown as 'Uncertain'.")
    p.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"),
                   help="Inference device (default cpu).")
    p.add_argument("--display-width", type=int, default=None,
                   help="Display window width in px (0 = original size).")
    p.add_argument("--detector-scale", type=float, default=None,
                   help="Downscale factor (0,1] applied before face detection, e.g. 0.5.")
    p.add_argument("--min-face", type=int, default=None,
                   help="Minimum face size in original pixels.")
    p.add_argument("--skip-frames", type=int, default=None,
                   help="Process 1 frame then reuse results for N frames.")
    p.add_argument("--save-output", default=None,
                   help="Output path (video .mp4, or image path for image mode).")
    p.add_argument("--config", default=None,
                   help=f"YAML config path (default: {DEFAULT_CONFIG_PATH.name} if exists).")
    p.add_argument("--no-display", action="store_true",
                   help="Do not open a window (headless / batch processing).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = build_settings(args)
    if not 0.0 <= settings.confidence <= 1.0:
        raise SystemExit("[ERROR] --confidence must be between 0 and 1.")
    if settings.skip_frames < 0:
        raise SystemExit("[ERROR] --skip-frames must be >= 0.")
    kind, value = parse_source(settings.source)

    print(f"[INFO] Source: {settings.source}  ({kind})")
    print(f"[INFO] Model: {settings.model}  weights: {settings.weights}  "
          f"device: {settings.device}")

    try:
        predictor = create_predictor(settings.model, settings.weights, settings.device)
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 2
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        return 2

    detector = FaceDetector(
        scale_factor=settings.scale_factor,
        min_neighbors=settings.min_neighbors,
        min_face_size=settings.min_face,
        margin=settings.margin,
        detect_scale=settings.detector_scale,
    )

    if kind == "image":
        return run_image(str(value), settings, detector, predictor)
    return run_stream(kind, value, settings, detector, predictor)


if __name__ == "__main__":
    sys.exit(main())
