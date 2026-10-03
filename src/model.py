"""Submission-compatible model definitions.

The project keeps the full implementations under src/models/. This file
provides the simple src/model.py entry point requested by the lab handout.
"""

from __future__ import annotations

from src.models.fer_attention_lite import FERAttentionLite
from src.models.mobilenet_v3_small import build_mobilenet_v3_small

__all__ = [
    "FERAttentionLite",
    "build_mobilenet_v3_small",
]
