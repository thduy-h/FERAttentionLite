"""Face detection.

`BaseFaceDetector` defines the interface; `HaarFaceDetector` is the default
OpenCV Haar Cascade implementation. Add MediaPipe/DNN detectors later by
subclassing `BaseFaceDetector` -- the app only uses `detect(frame)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


def expand_and_clamp(
    box: BBox,
    margin: float,
    frame_w: int,
    frame_h: int,
) -> BBox:
    """Expand a box by `margin` (fraction of w/h on each side), clamp to frame."""
    x, y, w, h = box
    dx, dy = int(round(w * margin)), int(round(h * margin))
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(frame_w, x + w + dx)
    y2 = min(frame_h, y + h + dy)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def crop_face(frame: np.ndarray, box: BBox) -> Optional[np.ndarray]:
    """Return the BGR crop for `box`, or None if empty."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    crop = frame[y:y + h, x:x + w]
    return crop if crop.size else None


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Return list of (x, y, w, h) boxes in frame coordinates."""


class HaarFaceDetector(BaseFaceDetector):
    """OpenCV Haar Cascade frontal face detector.

    Args:
        scale_factor:   cascade scaleFactor (>1.0).
        min_neighbors:  cascade minNeighbors.
        min_face_size:  minimum face side in *original* frame pixels.
        margin:         extra border around each face (fraction of w/h).
        detect_scale:   downscale factor applied before detection for speed
                        (1.0 = full res, 0.5 = half res). Boxes are mapped
                        back to original coordinates.
        cascade_path:   optional custom cascade XML.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: int = 40,
        margin: float = 0.10,
        detect_scale: float = 1.0,
        cascade_path: Optional[str] = None,
    ) -> None:
        if cascade_path is None:
            cascade_path = str(
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            )
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade: {cascade_path}. "
                "Install opencv-python 4.x (OpenCV 5 wheels no longer ship cascades)."
            )
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be > 1.0")
        if not 0.0 < detect_scale <= 1.0:
            raise ValueError("detect_scale must be in (0, 1]")
        if min_neighbors < 0:
            raise ValueError("min_neighbors must be >= 0")
        if min_face_size < 1:
            raise ValueError("min_face_size must be >= 1")
        if margin < 0:
            raise ValueError("margin must be >= 0")

        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_face_size = int(min_face_size)
        self.margin = float(margin)
        self.detect_scale = float(detect_scale)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        if frame is None or frame.size == 0:
            return []
        frame_h, frame_w = frame.shape[:2]

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = self.detect_scale
        if s < 1.0:
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)

        min_side = max(1, int(round(self.min_face_size * s)))
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(min_side, min_side),
        )

        boxes: List[BBox] = []
        for (x, y, w, h) in faces:
            box = (
                int(round(x / s)),
                int(round(y / s)),
                int(round(w / s)),
                int(round(h / s)),
            )
            box = expand_and_clamp(box, self.margin, frame_w, frame_h)
            if box[2] > 0 and box[3] > 0:
                boxes.append(box)
        # Sort left-to-right for stable drawing order.
        boxes.sort(key=lambda b: b[0])
        return boxes


# Default detector name used by the app.
FaceDetector = HaarFaceDetector
