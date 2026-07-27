#!/usr/bin/env python3
"""Build the audited single-class maize-leaf segmentation candidate pool."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT, get_output_root, get_project_data_root
from src.data.segmentation_audit import SEED
from src.data.segmentation_consolidation import build_segmentation_consolidation


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser with centralized project paths."""
    project_data = get_project_data_root()
    output = get_output_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=project_data / "leaf_detection" / "external_sources",
    )
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=project_data / "leaf_detection" / "pilot",
    )
    parser.add_argument(
        "--eda-root",
        type=Path,
        default=output / "leaf_detection" / "external_sources_eda",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_data / "leaf_detection" / "detector_dataset",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=output / "leaf_detection" / "detector_dataset_consolidation",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "dataset.yaml",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> None:
    """Consolidate candidates without training, downloads, or source mutation."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = build_segmentation_consolidation(
            args.external_root,
            args.pilot_root,
            args.eda_root,
            args.dataset_root,
            args.report_root,
            args.config,
            seed=args.seed,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    counts = summary["counts"]
    print(f"Imágenes consideradas: {counts['images_considered']}")
    print(f"Imágenes incluidas: {counts['images_included']}")
    print(f"Anotaciones incluidas: {counts['annotations_included']}")
    print(f"Recuperadas desde COCO: {counts['annotations_recovered']}")
    print(f"Revisión manual: {counts['manual_review_rows']}")
    print(f"Fugas contra piloto: {counts['pilot_leakage']}")
    print("Splits creados: no")
    print("Entrenamiento: no")
    print("Pesos descargados: no")


if __name__ == "__main__":
    main()
