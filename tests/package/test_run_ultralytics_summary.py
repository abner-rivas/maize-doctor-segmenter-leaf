"""Unit tests for the JSON-safe summary helpers of the cloud runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_ultralytics",
        PROJECT_ROOT / "cloud_training" / "run_ultralytics.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_ultralytics", module)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


def test_to_serializable_handles_tensors_scalars_and_none() -> None:
    payload = {
        "tensor": runner.to_serializable(_FakeTensor([1.5, 2.0])),
        "none": runner.to_serializable(None),
        "nested": runner.to_serializable([_FakeTensor([0.25]), 3, "x"]),
        "fallback": runner.to_serializable(object()),
    }
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["tensor"] == [1.5, 2.0]
    assert decoded["none"] is None
    assert decoded["nested"] == [[0.25], 3, "x"]
    assert isinstance(decoded["fallback"], str)


def test_resolve_trainer_prefers_model_trainer_over_metrics() -> None:
    trainer = SimpleNamespace(args=SimpleNamespace(batch=16), loss_items=_FakeTensor([0.1]))
    model = SimpleNamespace(trainer=trainer)
    metrics = SimpleNamespace()
    assert runner.resolve_trainer(model, metrics) is trainer
    selected = getattr(getattr(runner.resolve_trainer(model, metrics), "args", None), "batch", -1)
    assert selected == 16


def test_resolve_trainer_falls_back_to_result_then_none() -> None:
    trainer = SimpleNamespace(args=SimpleNamespace(batch=8))
    model = SimpleNamespace(trainer=None)
    metrics = SimpleNamespace(trainer=trainer)
    assert runner.resolve_trainer(model, metrics) is trainer
    empty_model = SimpleNamespace(trainer=None)
    empty_metrics = SimpleNamespace()
    resolved = runner.resolve_trainer(empty_model, empty_metrics)
    assert resolved is None
    fallback_batch = getattr(getattr(resolved, "args", None), "batch", -1)
    assert fallback_batch == -1
