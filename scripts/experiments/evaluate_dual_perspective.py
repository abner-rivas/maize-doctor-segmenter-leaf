"""Run a read-only, paired full-image versus segmented-leaf experiment.

The script never trains, changes a split, or creates a derived image dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from scripts.experiments.compare_full_vs_manual_roi import (
    load_compatible_checkpoint,
    resolve_device,
)
from scripts.pipeline.predict import _build_leaf_processor
from src.config import PROJECT_ROOT, get_dataset_root, get_output_root
from src.data.leaf_pilot import (
    read_csv_rows,
    require_columns,
    sha256_file,
    write_csv_rows,
)
from src.data.loader import load_and_normalize_image
from src.data.transforms import CornTransformFactory
from src.evaluation.dual_perspective import (
    DualPerspectiveExperimentRecord,
    build_dual_perspective_summary,
)
from src.inference.classifier import classify_image
from src.inference.dual_perspective import (
    DOMAIN_SHIFT_WARNING,
    DualPerspectiveConfig,
    classify_dual_perspective,
)
from src.models import list_models

REQUIRED_MANIFEST_COLUMNS = ("image_path", "ground_truth", "environment")
CASE_COLUMNS = (
    "file",
    "ground_truth",
    "environment",
    "multi_leaf",
    "severe_fall_armyworm",
    "segmentation_available",
    "segmentation_status",
    "segmentation_reason",
    "full_image_prediction",
    "full_image_confidence",
    "segmented_leaf_prediction",
    "segmented_leaf_confidence",
    "agreement",
    "full_image_correct",
    "segmented_leaf_correct",
)


@dataclass(frozen=True)
class ManifestCase:
    image_path: Path
    ground_truth: str
    environment: str
    multi_leaf: bool
    severe_fall_armyworm: bool


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe la configuración: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("la configuración debe ser un mapping")
    return loaded


def _strict_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"", "false", "0", "no"}:
        return False
    if normalized in {"true", "1", "yes", "sí", "si"}:
        return True
    raise ValueError(f"{name} debe ser booleano, recibido={value!r}")


def _resolve_image_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    beside_manifest = manifest_path.parent / candidate
    if beside_manifest.is_file():
        return beside_manifest
    return get_dataset_root() / candidate


def read_experiment_manifest(path: Path) -> list[ManifestCase]:
    rows, columns = read_csv_rows(path)
    require_columns(columns, REQUIRED_MANIFEST_COLUMNS, "dual perspective manifest")
    cases: list[ManifestCase] = []
    for line_number, row in enumerate(rows, start=2):
        image_path = _resolve_image_path(path, row["image_path"].strip())
        if not image_path.is_file():
            raise FileNotFoundError(f"imagen inexistente en línea {line_number}: {image_path}")
        ground_truth = row["ground_truth"].strip()
        environment = row["environment"].strip()
        if not ground_truth or not environment:
            raise ValueError(f"ground_truth y environment son obligatorios en línea {line_number}")
        cases.append(
            ManifestCase(
                image_path=image_path.resolve(),
                ground_truth=ground_truth,
                environment=environment,
                multi_leaf=_strict_bool(row.get("multi_leaf", ""), "multi_leaf"),
                severe_fall_armyworm=_strict_bool(
                    row.get("severe_fall_armyworm", ""),
                    "severe_fall_armyworm",
                ),
            )
        )
    if not cases:
        raise ValueError("el manifest experimental está vacío")
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimento opt-in de imagen completa y hoja segmentada.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=list_models(), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--segmenter-device", default=None)
    parser.add_argument("--segmenter-checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "dataset.yaml",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k debe ser mayor que cero")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser mayor que cero")
    return args


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    cases = read_experiment_manifest(args.manifest)
    if args.limit is not None:
        cases = cases[: args.limit]
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"No se sobrescribe el output experimental: {output_dir}")

    cfg = _load_config(args.config)
    classes = [str(name) for name in cfg["dataset"]["classes"]]
    unknown = sorted({case.ground_truth for case in cases} - set(classes))
    if unknown:
        raise ValueError(f"ground truth fuera de config/dataset.yaml: {unknown}")
    device = resolve_device(args.device)
    checkpoint = load_compatible_checkpoint(
        args.checkpoint,
        args.model,
        cfg,
        device,
    )
    transform = CornTransformFactory(
        config_path=str(args.config),
        target_size=checkpoint.target_size,
    ).get_pipeline("inference")
    policy = DualPerspectiveConfig.from_mapping(cfg["leaf_detection"])
    leaf_processor = _build_leaf_processor(
        cfg=cfg,
        output_root=get_output_root(),
        checkpoint_override=args.segmenter_checkpoint,
        segmenter_device=args.segmenter_device,
        processing_profile=policy.segmented_profile,
        target_size=checkpoint.target_size,
    )

    def classifier(image: Image.Image):
        return classify_image(
            checkpoint.model,
            image,
            transform=transform,
            idx_to_class=checkpoint.idx_to_class,
            device=device,
            top_k=args.top_k,
        )

    output_dir.mkdir(parents=True)
    records: list[DualPerspectiveExperimentRecord] = []
    structured: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        debug_dir = (
            output_dir / "debug" / f"{index:04d}_{case.image_path.stem}"
            if args.save_debug
            else None
        )
        result = classify_dual_perspective(
            load_and_normalize_image(str(case.image_path)),
            classifier=classifier,
            leaf_processor=leaf_processor,
            config=policy,
            source_image=case.image_path,
            debug_dir=debug_dir,
        )
        record = DualPerspectiveExperimentRecord(
            filename=str(case.image_path),
            ground_truth=case.ground_truth,
            environment=case.environment,
            result=result,
            multi_leaf=case.multi_leaf,
            severe_fall_armyworm=case.severe_fall_armyworm,
        )
        records.append(record)
        structured.append(result.to_metadata())

    rows = [record.to_metadata() for record in records]
    write_csv_rows(output_dir / "cases.csv", rows, CASE_COLUMNS)
    (output_dir / "structured_results.json").write_text(
        json.dumps(structured, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = build_dual_perspective_summary(records, classes)
    segmenter_metadata = leaf_processor.segmenter.to_metadata()
    summary["provenance"] = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "classifier_model": args.model,
        "classifier_checkpoint": str(args.checkpoint.resolve()),
        "classifier_checkpoint_sha256": checkpoint.sha256,
        "classifier_summary": checkpoint.summary_path,
        "segmenter_checkpoint": segmenter_metadata.get("segmenter_checkpoint"),
        "segmenter_checkpoint_sha256": segmenter_metadata.get("segmenter_checkpoint_sha256"),
        "target_size": list(checkpoint.target_size),
        "device": str(device),
        "domain_shift_warning": DOMAIN_SHIFT_WARNING,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
