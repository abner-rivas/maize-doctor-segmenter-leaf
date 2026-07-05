from __future__ import annotations

import torch.nn as nn

from src.models.registry import MODEL_REGISTRY


def build_mobilenet_v3_large(
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """MobileNetV3 Large baseline for corn leaf classification."""
    import timm

    return timm.create_model(
        "mobilenetv3_large_100",
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )


def build_mobilenet_v3_small(
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """MobileNetV3 Small baseline for corn leaf classification."""
    import timm

    return timm.create_model(
        "mobilenetv3_small_100",
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )


MODEL_REGISTRY.register("mobilenet_v3_large", factory=build_mobilenet_v3_large)
MODEL_REGISTRY.register("mobilenet_v3_small", factory=build_mobilenet_v3_small)
