"""Import manual YOLO or CSV leaf boxes into a new pilot annotation manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.roi_manifest import import_manual_annotations


def build_parser() -> argparse.ArgumentParser:
    """Create the manual annotation importer CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--format", choices=("yolo", "csv"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area-ratio", type=float, default=0.15)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar explícitamente --output; nunca se activa por defecto",
    )
    return parser


def main() -> None:
    """Import annotations and print the resulting status counts."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = import_manual_annotations(
            args.pilot_manifest,
            args.annotations,
            args.format,
            args.output,
            args.min_area_ratio,
            overwrite=args.overwrite,
            image_root=args.image_root,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Anotaciones importadas: {summary['output']}")
    print(f"Estados: {summary['status_counts']}")
    for warning in summary["warnings"]:
        print(f"ADVERTENCIA: {warning}")


if __name__ == "__main__":
    main()
