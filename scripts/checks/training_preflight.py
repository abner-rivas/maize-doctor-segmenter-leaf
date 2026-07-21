"""Check whether a local or remote host is ready for baseline training, without training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.config import PROJECT_ROOT, get_dataset_root, get_output_root
from src.training.preflight import run_training_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "dataset.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["efficientnet_b0", "shufflenet_v2_x1_0", "efficientnet_lite0"],
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check-dataset", action="store_true")
    parser.add_argument("--check-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = get_output_root()
    report = run_training_preflight(
        project_root=PROJECT_ROOT,
        splits_dir=args.splits_dir or output_root / "splits" / "seed_42_baseline",
        dataset_root=args.dataset_root or get_dataset_root(),
        config_path=args.config,
        models=args.models,
        device=args.device,
        check_dataset=args.check_dataset,
        check_gpu=args.check_gpu,
        output_dir=args.output or output_root / "preflight",
        results_dir=args.results_dir or output_root / "baselines",
    )
    print(f"Preflight: {'LISTO' if report['ready'] else 'BLOQUEADO'}")
    print(f"Bloqueos: {len(report['blockers'])}; advertencias: {len(report['warnings'])}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
