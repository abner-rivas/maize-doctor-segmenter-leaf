#!/usr/bin/env python3
"""Calibrate the quality gate on the frozen human-reviewed audit only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import yaml

from src.evaluation.segmentation_gate_calibration import (
    calibrate_gate,
    evaluate_gate,
    load_reviewed_audit,
)
from src.segmentation.quality import SegmentationQualityGateConfig

DEFAULT_AUDIT = Path(
    "outputs/leaf_detection/validation_real_pipeline/reliability_gate_audit_v1/audit_metrics.csv"
)
DEFAULT_OUTPUT = Path("outputs/leaf_detection/validation_real_pipeline/quality_gate_calibration_v1")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--config", type=Path, default=Path("config/segmentation.yaml"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    quality_mapping = payload["segmentation"]["quality_gate"]
    baseline = SegmentationQualityGateConfig.from_mapping(quality_mapping)
    rows = load_reviewed_audit(args.audit)
    recommended, sweep = calibrate_gate(
        rows,
        baseline=baseline,
        minimum_precision=args.minimum_precision,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "threshold_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    summary = {
        "schema_version": 1,
        "method": "bounded_grid_on_frozen_human_reviewed_audit",
        "audit_path": str(args.audit.resolve()),
        "audit_images": len(rows),
        "minimum_reliability_precision": args.minimum_precision,
        "selection_rule": (
            "meet minimum precision; maximize reliable GOOD masks; then precision, "
            "coverage and minimum distance from the current gate"
        ),
        "baseline": evaluate_gate(rows, baseline),
        "recommended": recommended,
        "candidate_count": len(sweep),
        "limitations": [
            "The 42-image reviewed audit is small and is reused for selection and reporting.",
            "Thresholds must be confirmed on a larger, newly annotated external sample.",
            "The internal test split is not read by this calibration.",
        ],
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
