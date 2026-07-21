"""Audit configured, documented, and physical classes without changing the dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.data.class_audit import audit_dataset_classes, audit_exit_code


def build_parser() -> argparse.ArgumentParser:
    """Build the class audit command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config/dataset.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--documentation",
        type=Path,
        action="append",
        default=None,
        help="Markdown con tabla class/lab/real/total; puede repetirse",
    )
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only audit and return 2 when strict mismatch mode fails."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.dataset_root is None or args.output is None:
            from src.config import get_dataset_root, get_output_root

            dataset_root = args.dataset_root or get_dataset_root()
            output = args.output or get_output_root() / "dataset_audit"
        else:
            dataset_root = args.dataset_root
            output = args.output
        documentation = args.documentation or [
            Path("docs/es/cleanup-and-ordered/index.md")
        ]
        report = audit_dataset_classes(
            dataset_root,
            args.config,
            output,
            documentation_paths=documentation,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Auditoría escrita en: {output.resolve()}")
    print(f"Imágenes soportadas: {report['total_images']}")
    print(f"Discrepancias críticas: {report['critical_mismatch_count']}")
    print(f"Recomendación: {report['split_recommendation']}")
    return audit_exit_code(report, args.fail_on_mismatch)


if __name__ == "__main__":
    raise SystemExit(main())
