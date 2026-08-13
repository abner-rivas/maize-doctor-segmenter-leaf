"""Build the final manual ROI manifest from an imported annotation manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.roi_manifest import build_roi_manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the final ROI manifest builder CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imported-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Generate the final CSV without transforming source images."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        rows = build_roi_manifest(
            args.imported_manifest,
            args.output,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Manifiesto ROI creado: {args.output.resolve()} ({rows} filas)")


if __name__ == "__main__":
    main()
