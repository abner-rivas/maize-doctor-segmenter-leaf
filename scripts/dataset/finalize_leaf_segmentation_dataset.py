#!/usr/bin/env python3
"""Rebuild and publish the definitive segmentation pool after human review."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT, get_output_root, get_project_data_root
from src.data.segmentation_audit import SEED
from src.data.segmentation_finalization import finalize_segmentation_dataset


def build_parser() -> argparse.ArgumentParser:
    """Create centralized-path CLI options."""
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
    """Finalize only after every source rebuild and validation succeeds."""
    args = build_parser().parse_args()
    result = finalize_segmentation_dataset(
        args.external_root,
        args.pilot_root,
        args.eda_root,
        args.dataset_root,
        args.report_root,
        args.config,
        seed=args.seed,
    )
    lock = result["lock"]
    print(f"Estado: {lock['status']}")
    print(f"Imágenes: {lock['total_images']}")
    print(f"Máscaras: {lock['total_masks']}")
    print(f"Fingerprint: {lock['global_fingerprint']['sha256']}")
    print("Splits creados: no")
    print("Entrenamiento: no")


if __name__ == "__main__":
    main()
