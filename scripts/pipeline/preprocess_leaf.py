#!/usr/bin/env python3
"""Generate an auditable leaf-mask preprocessing bundle for one image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT, get_output_root
from src.data.loader import load_and_normalize_image
from src.preprocessing.segmented_leaf_processor import (
    MASK_BLACK,
    SUPPORTED_MASK_PROFILES,
    SegmentedLeafProcessor,
    mask_processor_config_from_mapping,
)
from src.segmentation.leaf_segmenter import UltralyticsLeafSegmenter


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Configuración inválida: {path}")
    return payload


def _debug_id(path: Path) -> str:
    suffix = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]
    return f"{path.stem}-{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--profile",
        choices=sorted(SUPPORTED_MASK_PROFILES),
        default=MASK_BLACK,
    )
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=get_output_root() / "leaf_preprocessing_debug",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "dataset.yaml",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = _load_config(args.config)
    leaf_cfg = cfg["leaf_detection"]
    segmentation_cfg = leaf_cfg["segmentation"]
    checkpoint = args.checkpoint or (
        get_output_root() / str(segmentation_cfg["checkpoint"])
    )
    confidence = (
        float(args.confidence)
        if args.confidence is not None
        else float(leaf_cfg["confidence_threshold"])
    )
    iou = (
        float(args.iou)
        if args.iou is not None
        else float(segmentation_cfg["iou_threshold"])
    )
    target_size = tuple(int(value) for value in cfg["dataset"]["target_size"])
    processor_config = mask_processor_config_from_mapping(
        leaf_cfg,
        processing_profile=args.profile,
        confidence_threshold=confidence,
        target_size=(target_size[0], target_size[1]),
    )
    segmenter = UltralyticsLeafSegmenter(
        checkpoint,
        image_size=int(segmentation_cfg["image_size"]),
        confidence_threshold=confidence,
        iou_threshold=iou,
        max_detections=int(segmentation_cfg["max_detections"]),
        device=args.device,
        expected_version=str(segmentation_cfg["ultralytics_version"]),
    )
    image = load_and_normalize_image(str(args.image))
    output = args.output_root / _debug_id(args.image)
    result = SegmentedLeafProcessor(segmenter, processor_config).process(
        image,
        source_image=args.image,
        debug_dir=output,
    )
    print(json.dumps(result.to_metadata(), indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Debug: {output.resolve()}")


if __name__ == "__main__":
    main()
