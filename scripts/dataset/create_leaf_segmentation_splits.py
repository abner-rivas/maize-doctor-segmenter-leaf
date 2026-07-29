#!/usr/bin/env python3
"""Create reproducible, group-aware train/val/test splits for leaf segmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from src.config import get_output_root, get_project_data_root
from src.data.segmentation_split import (
    DEFAULT_PERCEPTUAL_THRESHOLD,
    SPLITS,
    apply_preserved_split_assignments,
    assign_groups_to_splits,
    build_split_groups,
    clone_records,
    load_split_records,
    materialize_split,
    verify_parent_dataset,
    write_dataset_yaml,
    write_split_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the safe split-only command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=get_project_data_root() / "leaf_detection" / "detector_dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=get_output_root() / "leaf_detection" / "detector_dataset_splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--materialization",
        choices=("copy", "hardlink"),
        default="copy",
        help=(
            "Materialización de los splits definitivos. Las dos corridas "
            "temporales de reproducibilidad siempre usan hardlink."
        ),
    )
    parser.add_argument(
        "--perceptual-threshold",
        type=int,
        default=DEFAULT_PERCEPTUAL_THRESHOLD,
    )
    return parser


def _deterministic_files(root: Path) -> dict[str, str]:
    relative_paths = (
        "manifests/split_manifest.csv",
        "manifests/split_groups.csv",
        "manifests/split_summary.json",
        "manifests/split_fingerprints.json",
        "manifests/split_lock.json",
        "dataset.yaml",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for split in SPLITS:
        names = sorted(path.name for path in (root / "images" / split).iterdir())
        result[f"assignment/{split}"] = hashlib.sha256(
            "\n".join(names).encode()
        ).hexdigest()
    return result


def _preserved_assignments(dataset_root: Path) -> dict[str, str] | None:
    manifest = dataset_root / "manifests" / "split_manifest.csv"
    if not manifest.is_file():
        return None
    import csv

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1155 or any(
        not row.get("filename") or row.get("split") not in SPLITS
        for row in rows
    ):
        raise RuntimeError(f"No se puede preservar una asignación inválida: {manifest}")
    assignments = {row["filename"]: row["split"] for row in rows}
    if len(assignments) != len(rows):
        raise RuntimeError(f"Hay filenames repetidos en {manifest}")
    return assignments


def _assign(
    groups,
    *,
    assignments: dict[str, str] | None,
    seed: int,
    ratios: dict[str, float],
) -> None:
    if assignments is None:
        assign_groups_to_splits(groups, seed=seed, ratios=ratios)
    else:
        apply_preserved_split_assignments(groups, assignments)


def _run_once(
    source_records,
    parent_lock,
    *,
    dataset_root: Path,
    report_root: Path,
    seed: int,
    ratios: dict[str, float],
    materialization: str,
    perceptual_threshold: int,
    pilot_root: Path,
    preserved_assignments: dict[str, str] | None,
) -> dict[str, str]:
    records = clone_records(source_records)
    groups = build_split_groups(records, perceptual_threshold=perceptual_threshold)
    _assign(
        groups,
        assignments=preserved_assignments,
        seed=seed,
        ratios=ratios,
    )
    materialize_split(records, dataset_root, materialization=materialization)
    write_dataset_yaml(dataset_root)
    write_split_artifacts(
        records,
        groups,
        dataset_root=dataset_root,
        output_root=report_root,
        parent_lock=parent_lock,
        seed=seed,
        ratios=ratios,
        perceptual_threshold=perceptual_threshold,
        pilot_root=pilot_root,
        reproducibility={"passed": True, "mode": "temporary_self_check"},
        render_visuals=False,
    )
    return _deterministic_files(dataset_root)


def main() -> None:
    """Verify the frozen parent, reproduce twice, then publish validated splits."""
    args = build_parser().parse_args()
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    parent_lock = verify_parent_dataset(args.dataset_root)
    source_records = load_split_records(args.dataset_root)
    preserved_assignments = _preserved_assignments(args.dataset_root)
    temporary_runs: list[dict[str, str]] = []
    # Las dos corridas de reproducibilidad sólo comparan asignaciones y
    # manifiestos: materializan por hardlink (con fallback a copia si el
    # sistema de archivos lo impide) para no copiar el dataset dos veces.
    # El directorio temporal vive junto al dataset para que el hardlink sea
    # posible en el mismo sistema de archivos.
    with tempfile.TemporaryDirectory(
        prefix=".tmp-leaf-segmentation-splits-",
        dir=args.dataset_root.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        for index in (1, 2):
            run_root = temporary_root / f"run_{index}"
            dataset_root = run_root / "detector_dataset"
            report_root = run_root / "outputs"
            (dataset_root / "manifests").mkdir(parents=True)
            temporary_runs.append(
                _run_once(
                    source_records,
                    parent_lock,
                    dataset_root=dataset_root,
                    report_root=report_root,
                    seed=args.seed,
                    ratios=ratios,
                    materialization="hardlink",
                    perceptual_threshold=args.perceptual_threshold,
                    pilot_root=args.dataset_root.parent / "pilot",
                    preserved_assignments=preserved_assignments,
                )
            )
        reproducibility = {
            "schema_version": 1,
            "seed": args.seed,
            "runs": 2,
            "assignments_identical": all(
                temporary_runs[0][key] == temporary_runs[1][key]
                for key in temporary_runs[0]
                if key.startswith("assignment/")
            ),
            "manifests_identical": all(
                temporary_runs[0][key] == temporary_runs[1][key]
                for key in temporary_runs[0]
                if key.startswith("manifests/")
            ),
            "fingerprints_identical": (
                temporary_runs[0]["manifests/split_fingerprints.json"]
                == temporary_runs[1]["manifests/split_fingerprints.json"]
            ),
            "assignment_mode": (
                "preserved_previous_manifest"
                if preserved_assignments is not None
                else "deterministic_group_balance"
            ),
            "file_sha256_run_1": temporary_runs[0],
            "file_sha256_run_2": temporary_runs[1],
        }
        reproducibility["passed"] = all(
            (
                reproducibility["assignments_identical"],
                reproducibility["manifests_identical"],
                reproducibility["fingerprints_identical"],
            )
        )
    if not reproducibility["passed"]:
        raise RuntimeError("La reconstrucción doble no fue determinista")

    records = clone_records(source_records)
    groups = build_split_groups(
        records, perceptual_threshold=args.perceptual_threshold
    )
    _assign(
        groups,
        assignments=preserved_assignments,
        seed=args.seed,
        ratios=ratios,
    )
    materialize_split(records, args.dataset_root, materialization=args.materialization)
    write_dataset_yaml(args.dataset_root)
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    lock = write_split_artifacts(
        records,
        groups,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        parent_lock=parent_lock,
        seed=args.seed,
        ratios=ratios,
        perceptual_threshold=args.perceptual_threshold,
        pilot_root=args.dataset_root.parent / "pilot",
        reproducibility=reproducibility,
        render_visuals=True,
    )
    print(json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
