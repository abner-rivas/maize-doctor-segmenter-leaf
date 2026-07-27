#!/usr/bin/env python3
"""Regenerate human-review previews from immutable source annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_output_root, get_project_data_root
from src.data.segmentation_audit import sha256_file
from src.data.segmentation_review import dataset_fingerprint
from src.data.segmentation_review_preview import generate_review_previews


def build_parser() -> argparse.ArgumentParser:
    """Create centralized path options for preview-only regeneration."""
    project_data = get_project_data_root()
    outputs = get_output_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=project_data / "leaf_detection" / "external_sources",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_data / "leaf_detection" / "detector_dataset",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=outputs / "leaf_detection" / "detector_dataset_consolidation",
    )
    return parser


def main() -> None:
    """Render previews while proving protected inputs remain byte-identical."""
    args = build_parser().parse_args()
    manifests = args.dataset_root / "manifests"
    protected = [
        manifests / "mandatory_visual_review.csv",
        manifests / "manual_review.csv",
        manifests / "dataset_lock.json",
    ]
    before_hashes = {
        str(path): sha256_file(path) for path in protected if path.is_file()
    }
    before_dataset = dataset_fingerprint(args.dataset_root)
    summary = generate_review_previews(
        args.external_root,
        manifests,
        args.dataset_root / "previews",
        args.report_root / "review_preview_validation.json",
    )
    after_hashes = {
        str(path): sha256_file(path) for path in protected if path.is_file()
    }
    after_dataset = dataset_fingerprint(args.dataset_root)
    if before_hashes != after_hashes:
        raise RuntimeError("Se modificó un manifiesto humano o dataset_lock")
    if before_dataset != after_dataset:
        raise RuntimeError("Se modificó all/ o un manifiesto del dataset")
    print(f"Estado: {summary['global_status']}")
    print(f"Casos únicos: {summary['total_unique_cases']}")
    print(f"Previews: {summary['previews_generated']}")
    print(f"Fuentes: {summary['geometry_sources']}")
    print(f"Estados: {summary['render_statuses']}")


if __name__ == "__main__":
    main()
