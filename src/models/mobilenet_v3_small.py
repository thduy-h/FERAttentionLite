"""MobileNetV3-Small transfer-learning model for FER-2013."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 7


def build_mobilenet_v3_small(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
) -> nn.Module:
    """Build the exact M2 architecture used by the Kaggle notebook.

    During training, use pretrained=True to initialize from ImageNet.
    During inference, pretrained=False is correct because the trained FER
    checkpoint fully supplies the learned weights.
    """
    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
