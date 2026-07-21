"""Validate a manual ROI manifest, leakage, coverage, and optional previews."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from src.preprocessing.leaf_processor import LeafProcessorConfig
from src.preprocessing.roi_manifest import validate_roi_manifest


def _parse_padding(values: Sequence[int]) -> int | tuple[int, int, int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 3:
        return values[0], values[1], values[2]
    raise ValueError("--padding-value acepta uno o tres enteros")


def build_parser() -> argparse.ArgumentParser:
    """Create the ROI manifest validator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/dataset.yaml"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--min-area-ratio", type=float, default=None)
    parser.add_argument("--preview-samples", type=int, default=0)
    parser.add_argument("--preview-output", type=Path, default=None)
    parser.add_argument("--preview-seed", type=int, default=42)
    parser.add_argument("--margin-ratio", type=float, default=None)
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--padding-value", type=int, nargs="+", default=None)
    return parser


def main() -> None:
    """Validate the manifest and exit nonzero only for execution/configuration failures."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        if not args.config.is_file():
            raise FileNotFoundError(f"Configuración inexistente: {args.config}")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        leaf_config = config.get("leaf_detection", {})
        min_area_ratio = (
            args.min_area_ratio
            if args.min_area_ratio is not None
            else float(leaf_config.get("min_area_ratio", 0.15))
        )
        margin_ratio = (
            args.margin_ratio
            if args.margin_ratio is not None
            else float(leaf_config.get("margin_ratio", 0.08))
        )
        target_size = tuple(args.target_size or config["dataset"]["target_size"])
        padding = _parse_padding(
            args.padding_value
            if args.padding_value is not None
            else [int(leaf_config.get("padding_value", 0))]
        )
        if args.preview_samples < 0:
            raise ValueError("--preview-samples no puede ser negativo")
        if args.preview_samples and args.preview_output is None:
            raise ValueError("--preview-output es obligatorio cuando --preview-samples > 0")
        processor_config = LeafProcessorConfig(
            margin_ratio=margin_ratio,
            min_area_ratio=min_area_ratio,
            target_size=target_size,
            padding_value=padding,
            fallback="reject",
            preserve_aspect_ratio=True,
        )
        summary = validate_roi_manifest(
            args.roi_manifest,
            args.output,
            valid_classes=set(config["dataset"]["classes"]),
            min_area_ratio=min_area_ratio,
            image_root=args.image_root,
            preview_samples=args.preview_samples,
            preview_output=args.preview_output,
            preview_seed=args.preview_seed,
            processor_config=processor_config,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Validación completada: {args.output.resolve()}")
    print(f"Filas válidas: {summary['valid_rows']} / {summary['total_rows']}")
    print(f"Vistas previas: {summary['preview_generated']}")


if __name__ == "__main__":
    main()
