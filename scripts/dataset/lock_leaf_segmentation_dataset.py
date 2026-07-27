#!/usr/bin/env python3
"""Validate manual reviews and write the segmentation dataset gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import get_output_root, get_project_data_root
from src.data.segmentation_review import (
    build_dataset_lock,
    read_review_manifest,
    validate_review_manifests,
    write_applied_review_report,
    write_dataset_lock,
    write_reannotation_queue,
)


def build_parser() -> argparse.ArgumentParser:
    """Create centralized-path CLI options."""
    dataset_root = get_project_data_root() / "leaf_detection" / "detector_dataset"
    outputs = get_output_root() / "leaf_detection"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=dataset_root)
    parser.add_argument(
        "--eda-summary",
        type=Path,
        default=outputs / "external_sources_eda" / "summary.json",
    )
    parser.add_argument(
        "--consolidation-summary",
        type=Path,
        default=outputs / "detector_dataset_consolidation" / "summary.json",
    )
    return parser


def main() -> None:
    """Write a blocked lock until human reviews and a source rebuild are complete."""
    args = build_parser().parse_args()
    manifests = args.dataset_root / "manifests"
    mandatory = read_review_manifest(manifests / "mandatory_visual_review.csv")
    general = read_review_manifest(manifests / "manual_review.csv")
    review_summary = validate_review_manifests(mandatory, general)
    applied = [
        *review_summary["approved"],
        *review_summary["excluded"],
        *review_summary["reannotation"],
    ]
    write_applied_review_report(
        manifests / "review_decisions_applied.csv",
        sorted(applied, key=lambda row: row["review_key"]),
    )
    write_reannotation_queue(
        manifests / "reannotation_queue.csv",
        review_summary["reannotation"],
    )
    consolidation_summary = json.loads(
        args.consolidation_summary.read_text(encoding="utf-8")
    )
    eda_summary = json.loads(args.eda_summary.read_text(encoding="utf-8"))
    lock = build_dataset_lock(
        args.dataset_root,
        consolidation_summary,
        eda_summary,
        review_summary,
        decisions_applied_from_sources=False,
    )
    output = manifests / "dataset_lock.json"
    write_dataset_lock(output, lock)
    print(f"Estado: {lock['status']}")
    print(f"Revisiones obligatorias pendientes: {lock['review_summary']['mandatory_pending']}")
    print(f"Revisiones únicas pendientes: {len(lock['pending_reviews'])}")
    print(f"Lock: {output}")


if __name__ == "__main__":
    main()
