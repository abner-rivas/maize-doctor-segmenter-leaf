"""Tests for deterministic segmentation-ablation inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from src.training.segmentation_experiments import materialize_source_balanced_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_source_balancing_oversamples_train_without_test_or_pilot(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "manifests").mkdir(parents=True)
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    rows = []
    for source, count in (("large", 4), ("corn", 2)):
        for index in range(count):
            filename = f"{source}_{index}.jpg"
            relative = f"images/train/{filename}"
            (dataset / relative).write_bytes(b"image")
            rows.append(
                {
                    "split": "train",
                    "filename": filename,
                    "source_dataset": source,
                    "materialized_image_path": relative,
                }
            )
    rows.append(
        {
            "split": "test",
            "filename": "retained.jpg",
            "source_dataset": "corn",
            "materialized_image_path": "images/test/retained.jpg",
        }
    )
    with (dataset / "manifests" / "split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = materialize_source_balanced_dataset(dataset, tmp_path / "output")
    train_lines = Path(str(metadata["train_list"])).read_text().splitlines()
    generated_yaml = yaml.safe_load(Path(str(metadata["dataset_yaml"])).read_text())

    assert metadata["input_source_counts"] == {"corn": 2, "large": 4}
    assert metadata["output_source_counts"] == {"corn": 4, "large": 4}
    assert len(train_lines) == 8
    assert all("retained" not in line for line in train_lines)
    assert "test" not in generated_yaml
    assert metadata["pilot_included"] is False


def test_all_experiment_profiles_are_unique_and_keep_test_out_of_training() -> None:
    directory = PROJECT_ROOT / "cloud_training" / "configs" / "experiments"
    configs = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.yaml"))
        if path.name != "plan.yaml"
    ]
    assert len(configs) == 9
    assert len({config["name"] for config in configs}) == len(configs)
    assert all(config["task"] == "segment" for config in configs)
    assert all(config["deterministic"] is True for config in configs)
    assert {config["seed"] for config in configs} <= {7, 42, 1337}
    assert all("/test" not in str(config["data"]) for config in configs)
