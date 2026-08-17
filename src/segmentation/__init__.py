"""Leaf-segmentation inference interfaces."""

from src.segmentation.leaf_segmenter import (
    LeafInstance,
    LeafSegmenter,
    UltralyticsLeafSegmenter,
    instances_from_ultralytics_result,
    rasterize_instance_polygon,
)
from src.segmentation.quality import (
    SegmentationAssessment,
    SegmentationQualityGateConfig,
    SegmentationStatus,
    assess_segmentation,
    assess_segmentation_legacy,
)

__all__ = [
    "LeafInstance",
    "LeafSegmenter",
    "SegmentationAssessment",
    "SegmentationQualityGateConfig",
    "SegmentationStatus",
    "UltralyticsLeafSegmenter",
    "assess_segmentation",
    "assess_segmentation_legacy",
    "instances_from_ultralytics_result",
    "rasterize_instance_polygon",
]
