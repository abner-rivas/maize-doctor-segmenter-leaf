"""Build a reproducible manual-annotation pilot from an existing split CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.leaf_pilot import (
    SUPPORTED_COPY_MODES,
    SUPPORTED_SELECTION_STRATEGIES,
    VALID_ENVIRONMENTS,
    VALID_SPLITS,
    build_pilot,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the pilot builder command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directorio nuevo para el piloto; nunca se escribe dentro de DATASET_ROOT",
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        choices=VALID_ENVIRONMENTS,
        default=["real"],
    )
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--copy-mode", choices=SUPPORTED_COPY_MODES, default="copy")
    parser.add_argument(
        "--selection-strategy",
        choices=SUPPORTED_SELECTION_STRATEGIES,
        default="balanced",
    )
    parser.add_argument(
        "--priority-manifest",
        type=Path,
        default=None,
        help="CSV opcional con image_path y señales correct/pred_label/pred_prob",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--split-name", choices=VALID_SPLITS, default=None)
    return parser


def main() -> None:
    """Build the requested pilot and report generated artifact locations."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.dataset_root is None:
            from src.config import get_dataset_root

            dataset_root = get_dataset_root()
        else:
            dataset_root = args.dataset_root
        summary = build_pilot(
            args.split_csv,
            dataset_root,
            args.output,
            samples=args.samples,
            seed=args.seed,
            environments=args.environments,
            classes=args.classes,
            copy_mode=args.copy_mode,
            selection_strategy=args.selection_strategy,
            priority_manifest=args.priority_manifest,
            split_name=args.split_name,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Piloto creado: {summary['output_root']}")
    print(f"Imágenes seleccionadas: {summary['selected_samples']}")
    for warning in summary["warnings"]:
        print(f"ADVERTENCIA: {warning}")


if __name__ == "__main__":
    main()
