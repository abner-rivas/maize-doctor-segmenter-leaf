"""Group segmentation inference artifacts into a classifier-ready dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inference.classifier_dataset import (
    DEFAULT_ENVIRONMENT,
    SUPPORTED_STORAGE_MODES,
    export_classifier_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Por defecto: <inference-run>/classifier_dataset",
    )
    parser.add_argument(
        "--include-status",
        action="append",
        default=None,
        choices=("reliable", "uncertain"),
        help="Estado de segmentación que se incluirá; repetible (default: reliable)",
    )
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument(
        "--storage-mode",
        choices=SUPPORTED_STORAGE_MODES,
        default="hardlink",
        help="hardlink evita duplicar bytes; copy produce una copia física independiente",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_classifier_dataset(
        args.inference_run,
        output_dir=args.output,
        reliability_statuses=args.include_status or ("reliable",),
        environment=args.environment,
        storage_mode=args.storage_mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
