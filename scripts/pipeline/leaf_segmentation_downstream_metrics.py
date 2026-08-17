#!/usr/bin/env python3
"""Compute downstream segmentation metrics from frozen labels and predictions.

Never loads a model and never trains: consumes YOLO-seg prediction files (the
output of ``yolo ... save_txt=True``) so the metrics are reproducible on any
machine, with or without GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

from src.config import get_output_root, get_project_data_root
from src.evaluation.segmentation_downstream import (
    DEFAULT_MIN_AREA,
    DEFAULT_RASTER_SIZE,
    ROW_COLUMNS,
    evaluate_downstream,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=get_project_data_root() / "leaf_detection" / "detector_dataset",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Directorio con las etiquetas predichas en formato YOLO-seg",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=get_output_root() / "leaf_detection" / "downstream_metrics",
    )
    parser.add_argument("--raster-size", type=int, default=DEFAULT_RASTER_SIZE)
    parser.add_argument("--minimum-area", type=float, default=DEFAULT_MIN_AREA)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.split == "test":
        print(
            "AVISO: el test interno se evalúa una sola vez con la configuración "
            "congelada; no lo use para elegir hiperparámetros.",
        )
    rows, summary = evaluate_downstream(
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        split=args.split,
        raster_size=args.raster_size,
        minimum_area=args.minimum_area,
    )
    output = args.output_root / args.split
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cast(Any, rows))
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True, ensure_ascii=False))
    print(json.dumps(summary["by_source"], indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
