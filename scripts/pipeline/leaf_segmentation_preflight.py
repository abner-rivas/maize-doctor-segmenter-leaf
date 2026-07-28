#!/usr/bin/env python3
"""Run the leaf-segmentation training preflight without training or downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import get_output_root, get_project_data_root
from src.training.segmentation_preflight import run_segmentation_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=get_project_data_root() / "leaf_detection" / "detector_dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=get_output_root() / "leaf_detection" / "training_preflight",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    summary = run_segmentation_preflight(
        project_root=project_root,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
