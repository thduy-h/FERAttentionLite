"""Drawing helpers (bounding boxes, labels, status overlay, resizing)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]

COLOR_CONFIDENT: Color = (0, 200, 0)    # green
COLOR_UNCERTAIN: Color = (0, 165, 255)  # orange
COLOR_TEXT: Color = (255, 255, 255)
COLOR_STATUS_BG: Color = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_scale(frame: np.ndarray) -> Tuple[float, int]:
    """Font scale/thickness relative to frame width."""
    w = frame.shape[1]
    scale = max(0.45, min(1.0, w / 1000.0))
    thickness = 1 if scale < 0.7 else 2
    return scale, thickness


def draw_prediction(
    frame: np.ndarray,
    box: Sequence[int],
    label: str,
    confidence: float,
    threshold: float = 0.40,
) -> None:
    """Draw bbox + 'Label 92.4%' in place.

    Below threshold: shows 'Uncertain (Label) 38.2%' in orange.
    """
    x, y, w, h = [int(v) for v in box]
    uncertain = confidence < threshold
    color = COLOR_UNCERTAIN if uncertain else COLOR_CONFIDENT
    text = (
        f"Uncertain ({label}) {confidence * 100:.1f}%"
        if uncertain
        else f"{label} {confidence * 100:.1f}%"
    )

    scale, thick = _text_scale(frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, max(2, thick))

    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thick)
    ty = y - 6 if y - th - baseline - 6 >= 0 else y + h + th + 6
    tx = max(0, min(x, frame.shape[1] - tw - 6))  # keep label inside frame
    cv2.rectangle(
        frame,
        (tx, ty - th - baseline),
        (tx + tw + 6, ty + baseline),
        color,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (tx + 3, ty), FONT, scale, COLOR_TEXT, thick, cv2.LINE_AA)


def draw_status(
    frame: np.ndarray,
    fps: Optional[float],
    num_faces: int,
    model_name: str,
    extra: Optional[str] = None,
) -> None:
    """Top-left overlay: FPS / Faces / Model (+ optional extra line)."""
    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    lines.append(f"Faces: {num_faces}")
    lines.append(f"Model: {model_name}")
    if extra:
        lines.append(extra)

    scale, thick = _text_scale(frame)
    pad = 6
    line_h = int(cv2.getTextSize("Ag", FONT, scale, thick)[0][1] * 1.8)
    box_w = max(cv2.getTextSize(t, FONT, scale, thick)[0][0] for t in lines) + 2 * pad
    box_h = line_h * len(lines) + pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), COLOR_STATUS_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)

    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (pad, (i + 1) * line_h),
            FONT, scale, COLOR_TEXT, thick, cv2.LINE_AA,
        )


def resize_for_display(frame: np.ndarray, display_width: Optional[int]) -> np.ndarray:
    """Resize keeping aspect ratio so width == display_width (if set & smaller)."""
    if not display_width or display_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == display_width:
        return frame
    scale = display_width / float(w)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (display_width, int(round(h * scale))), interpolation=interp)
