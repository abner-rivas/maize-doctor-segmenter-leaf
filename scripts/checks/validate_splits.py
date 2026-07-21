"""Validate existing official splits without generating or changing them."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from src.data.split_audit import split_validation_exit_code, validate_splits


def build_parser() -> argparse.ArgumentParser:
    """Build the split validation command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config/dataset.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compare-dir", type=Path, default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def _default_splits_dir(config_path: Path, output_root: Path) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuración inexistente: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    relative = Path(config["paths"]["split_output_dir"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("paths.split_output_dir debe ser relativo a OUTPUT_ROOT")
    return output_root / f"{relative}_baseline"


def main(argv: Sequence[str] | None = None) -> int:
    """Run split validation and return a strict-mode status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.dataset_root is None or args.splits_dir is None:
            from src.config import get_dataset_root, get_output_root

            dataset_root = args.dataset_root or get_dataset_root()
            splits_dir = args.splits_dir or _default_splits_dir(
                args.config,
                get_output_root(),
            )
        else:
            dataset_root = args.dataset_root
            splits_dir = args.splits_dir
        output = args.output or splits_dir
        report = validate_splits(
            splits_dir,
            dataset_root,
            args.config,
            output,
            compare_dir=args.compare_dir,
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Validación escrita en: {output.resolve()}")
    print(f"Filas: {report['total_rows']}")
    print(f"Errores: {report['error_count']}")
    return split_validation_exit_code(report, args.fail_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
