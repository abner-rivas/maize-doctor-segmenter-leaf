"""Run the current Doctor Maíz leaf segmenter over every image in a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT, get_output_root
from src.inference.full_dataset import exact_command, run_full_dataset

DEFAULT_RUN_ID = "2026-08-27_updated_hf_dataset"
DEFAULT_OUTPUT = get_output_root() / "full_dataset_inference" / DEFAULT_RUN_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "segmentation.yaml",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--dataset-source",
        default="https://huggingface.co/datasets/daiv05/corn-leaf-diseases-pests-and-deficiencies",
    )
    parser.add_argument("--dataset-commit", default="")
    parser.add_argument("--source-repository", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--inventory-only", action="store_true")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--reuse-inventory", action="store_true")
    continuation.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_full_dataset(
        dataset_root=args.dataset,
        output_dir=args.output,
        config_path=args.config,
        metadata_path=args.metadata,
        dataset_source=args.dataset_source,
        dataset_commit=args.dataset_commit,
        source_repository=args.source_repository,
        device=args.device,
        limit=args.limit,
        progress_every=args.progress_every,
        inventory_only=args.inventory_only,
        reuse_inventory=args.reuse_inventory,
        resume=args.resume,
        command=exact_command(module="scripts.pipeline.infer_dataset"),
    )


if __name__ == "__main__":
    main()
