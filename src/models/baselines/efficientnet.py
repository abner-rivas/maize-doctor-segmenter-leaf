from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models

from src.models.registry import MODEL_REGISTRY


def build_efficientnet_b0(
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = tv_models.efficientnet_b0(weights=weights, **kwargs)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_lite0(
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    import timm

    return timm.create_model(
        "efficientnet_lite0",
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )


def build_efficientnet_b4(
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """EfficientNet-B4 baseline for corn leaf disease classification."""
    import timm

    return timm.create_model(
        "efficientnet_b4",
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )


MODEL_REGISTRY.register("efficientnet_b0", factory=build_efficientnet_b0)
MODEL_REGISTRY.register("efficientnet_lite0", factory=build_efficientnet_lite0)
MODEL_REGISTRY.register("efficientnet_b4", factory=build_efficientnet_b4)
