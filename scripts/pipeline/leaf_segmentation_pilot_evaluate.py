#!/usr/bin/env python3
"""Run authorized qualitative evaluation on the externally held-out pilot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ultralytics import YOLO

from src.config import PROJECT_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    configured_output = Path(
        os.getenv("LEAF_SEGMENTATION_OUTPUT", "outputs/leaf_detection")
    )
    output = (
        configured_output
        if configured_output.is_absolute()
        else PROJECT_ROOT / configured_output
    )
    device = os.getenv("SEGMENTATION_DEVICE", "0")
    test_summary = output / "segmenter_evaluation" / "test_summary.json"
    if not test_summary.is_file() or json.loads(
        test_summary.read_text(encoding="utf-8")
    ).get("status") != "passed":
        raise SystemExit("El test interno debe estar aprobado antes del piloto")
    checkpoint = (
        output / "segmenter" / "yolo26n_seg_baseline" / "weights" / "best.pt"
    )
    if not checkpoint.is_file():
        raise SystemExit(f"Falta checkpoint: {checkpoint}")
    pilot = PROJECT_ROOT / "data" / "leaf_detection" / "pilot" / "images"
    results = YOLO(str(checkpoint)).predict(
        source=[str(path) for path in sorted(pilot.iterdir()) if path.is_file()],
        imgsz=640,
        device=device,
        save=True,
        project=str(output / "pilot_external_evaluation"),
        name="yolo26n_seg_pilot_predictions",
        verbose=False,
    )
    summary = {
        "status": "predictions_generated",
        "evaluation_role": "external_held_out_pilot",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "images": len(results),
        "metrics_computed": False,
        "reason": "Pilot annotations use the historical ROI rule; qualitative review only",
    }
    summary_path = output / "pilot_external_evaluation" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
