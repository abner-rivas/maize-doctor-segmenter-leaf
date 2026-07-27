"""Prepare reproducible CVAT batches and a retained test for the leaf detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_dataset_root, get_project_data_root
from src.data.leaf_detector_dataset import build_detector_annotation_set


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset-preparation CLI without importing Ultralytics."""
    project_data = get_project_data_root()
    pilot = project_data / "leaf_detection" / "pilot"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=project_data / "splits" / "seed_42_baseline" / "train.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=project_data / "splits" / "seed_42_baseline" / "val.csv",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=pilot)
    parser.add_argument(
        "--imported-annotations",
        type=Path,
        default=pilot / "manifests" / "imported_annotations.csv",
    )
    parser.add_argument(
        "--cvat-xml",
        type=Path,
        default=pilot / "annotations" / "cvat" / "annotations.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_data / "leaf_detection" / "detector_dataset",
    )
    parser.add_argument("--train-count", type=int, default=350)
    parser.add_argument("--val-count", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-fraction", type=float, default=0.8)
    return parser


def main() -> None:
    """Create annotation batches; never train or load a YOLO model."""
    parser = build_parser()
    args = parser.parse_args()
    dataset_root = args.dataset_root or get_dataset_root()
    try:
        summary = build_detector_annotation_set(
            args.train_csv,
            args.val_csv,
            dataset_root,
            args.pilot_root,
            args.imported_annotations,
            args.cvat_xml,
            args.output,
            train_count=args.train_count,
            val_count=args.val_count,
            seed=args.seed,
            real_fraction=args.real_fraction,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Dataset de anotación creado: {args.output.resolve()}")
    print(f"Train pendiente: {summary['counts']['train']}")
    print(f"Val pendiente: {summary['counts']['val']}")
    print(f"Test anotado: {summary['counts']['test_annotated']}")
    print(f"Fugas: {not summary['leakage_zero']}")
    print("Entrenamiento: no")
    print("Descarga de pesos: no")


if __name__ == "__main__":
    main()
