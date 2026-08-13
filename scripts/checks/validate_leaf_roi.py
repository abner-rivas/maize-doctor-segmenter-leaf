"""Generate visual evidence for bbox, margin, crop, and letterbox processing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from src.data.loader import load_and_normalize_image
from src.preprocessing.leaf_roi import BoundingBox, crop_leaf_region, image_to_rgb
from src.preprocessing.roi_processor import (
    FALLBACK_CENTER_CROP,
    FALLBACK_ORIGINAL,
    FALLBACK_REJECT,
    LeafImageProcessor,
    LeafProcessingResult,
    LeafProcessorConfig,
)

OUTPUT_NAMES = (
    "01_original.jpg",
    "02_bbox_original.jpg",
    "03_bbox_with_margin.jpg",
    "04_crop.jpg",
    "05_letterbox.jpg",
    "metadata.json",
)

INTERPOLATIONS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _parse_padding(values: Sequence[int]) -> int | tuple[int, int, int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 3:
        return values[0], values[1], values[2]
    raise ValueError("--padding-value debe recibir un valor o tres canales RGB")


def _draw_bbox(
    image: Image.Image,
    bbox: BoundingBox | None,
    *,
    color: tuple[int, int, int],
    label: str,
) -> Image.Image:
    canvas = image_to_rgb(image)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        draw.rectangle(
            (x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)),
            outline=color,
            width=max(2, min(canvas.size) // 200),
        )
    text_box = draw.textbbox((5, 5), label, font=font)
    draw.rectangle(text_box, fill=(0, 0, 0))
    draw.text((5, 5), label, fill=color, font=font)
    return canvas


def _placeholder(size: tuple[int, int], message: str) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", (max(1, width), max(1, height)), (35, 35, 35))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), message, fill=(255, 100, 100), font=ImageFont.load_default())
    return canvas


def _crop_for_debug(image: Image.Image, result: LeafProcessingResult) -> Image.Image | None:
    if result.expanded_bbox is not None:
        return crop_leaf_region(image, result.expanded_bbox)
    if result.fallback_bbox is not None:
        return crop_leaf_region(image, result.fallback_bbox)
    return None


def _json_safe_bbox(values: Sequence[float]) -> list[float | str]:
    return [value if math.isfinite(value) else str(value) for value in values]


def _build_metadata(
    image_path: Path,
    bbox_received: Sequence[float],
    result: LeafProcessingResult,
    config: LeafProcessorConfig,
) -> dict[str, object]:
    metadata = result.to_metadata()
    return {
        "input_path": str(image_path.resolve()),
        "original_size": list(result.original_size),
        "bbox_received": _json_safe_bbox(bbox_received),
        "bbox_clipped": list(result.clipped_bbox) if result.clipped_bbox else None,
        "bbox_with_margin": list(result.expanded_bbox) if result.expanded_bbox else None,
        "area_ratio": result.detection_result.area_ratio,
        "margin_ratio": config.margin_ratio,
        "min_area_ratio": config.min_area_ratio,
        "crop_size": list(result.crop_size) if result.crop_size else None,
        "final_size": list(result.processed_size) if result.processed_size else None,
        "target_size_height_width": list(config.target_size),
        "padding_value": list(config.padding_value)
        if isinstance(config.padding_value, tuple)
        else config.padding_value,
        "fallback": config.fallback,
        "fallback_used": result.fallback_used,
        "preserve_aspect_ratio": config.preserve_aspect_ratio,
        "warnings": list(result.warnings),
        "processing": metadata,
    }


def _validate_output_is_safe(image_path: Path, output_dir: Path) -> None:
    input_resolved = image_path.resolve()
    for name in OUTPUT_NAMES:
        if (output_dir / name).resolve() == input_resolved:
            raise ValueError(
                f"La salida {name} sobrescribiría la imagen de entrada; use otro --output"
            )


def run_validation(
    image_path: Path,
    bbox: Sequence[float],
    output_dir: Path,
    config: LeafProcessorConfig,
    *,
    confidence: float = 1.0,
) -> dict[str, object]:
    """Process one image and persist the five debug stages plus JSON metadata."""
    _validate_output_is_safe(image_path, output_dir)
    image = load_and_normalize_image(image_path)
    processor = LeafImageProcessor(config)
    result = processor.process(image, bbox, confidence=confidence, source="manual")

    output_dir.mkdir(parents=True, exist_ok=True)
    original = image_to_rgb(image)
    original.save(output_dir / "01_original.jpg", quality=95)
    bbox_label = "bbox recibido / limitado"
    _draw_bbox(
        original,
        result.clipped_bbox,
        color=(255, 210, 0),
        label=bbox_label,
    ).save(output_dir / "02_bbox_original.jpg", quality=95)

    margin_box = result.expanded_bbox or result.fallback_bbox
    margin_label = "bbox con margen" if result.expanded_bbox else "región de fallback"
    _draw_bbox(
        original,
        margin_box,
        color=(0, 255, 100) if result.expanded_bbox else (255, 140, 0),
        label=margin_label,
    ).save(output_dir / "03_bbox_with_margin.jpg", quality=95)

    crop = _crop_for_debug(image, result)
    if crop is None:
        crop = _placeholder(image.size, "sin recorte: fallback reject")
    crop.save(output_dir / "04_crop.jpg", quality=95)

    final_image = result.processed_image
    if final_image is None:
        target_height, target_width = config.target_size
        final_image = _placeholder((target_width, target_height), "salida rechazada")
    final_image.save(output_dir / "05_letterbox.jpg", quality=95)

    metadata = _build_metadata(image_path, bbox, result, config)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Build the manual ROI validation command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Imagen fuente")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("X1", "Y1", "X2", "Y2"),
    )
    parser.add_argument("--margin-ratio", type=float, default=0.08)
    parser.add_argument("--min-area-ratio", type=float, default=0.15)
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=(640, 640),
        metavar=("HEIGHT", "WIDTH"),
        help="Tamaño final con convención (alto, ancho)",
    )
    parser.add_argument(
        "--padding-value",
        type=int,
        nargs="+",
        default=(0,),
        metavar="CHANNEL",
        help="Un valor de gris o tres canales R G B",
    )
    parser.add_argument(
        "--fallback",
        choices=(FALLBACK_ORIGINAL, FALLBACK_CENTER_CROP, FALLBACK_REJECT),
        default=FALLBACK_ORIGINAL,
    )
    parser.add_argument("--center-crop-ratio", type=float, default=0.8)
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument(
        "--interpolation",
        choices=tuple(INTERPOLATIONS),
        default="bilinear",
    )
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="Desactiva letterbox; sólo para comprobar el modo preserve_aspect_ratio=false",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Run the CLI with specific, user-facing errors."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        padding = _parse_padding(args.padding_value)
        config = LeafProcessorConfig(
            margin_ratio=args.margin_ratio,
            min_area_ratio=args.min_area_ratio,
            target_size=tuple(args.target_size),
            padding_value=padding,
            fallback=args.fallback,
            preserve_aspect_ratio=not args.stretch,
            center_crop_ratio=args.center_crop_ratio,
            resample=INTERPOLATIONS[args.interpolation],
        )
        metadata = run_validation(
            args.image,
            args.bbox,
            args.output,
            config,
            confidence=args.confidence,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Validación ROI completada: {args.output.resolve()}")
    print(f"Tamaño final: {metadata['final_size']} | fallback: {metadata['fallback_used']}")


if __name__ == "__main__":
    main()
