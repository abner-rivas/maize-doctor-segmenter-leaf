"""Write the deterministic remote-training package manifest (no ZIP is created)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.config import PROJECT_ROOT, get_output_root
from src.training.package_manifest import write_training_package_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or get_output_root() / "training_package_manifest.json"
    manifest = write_training_package_manifest(args.project_root, output)
    print(f"Manifiesto: {output.resolve()}")
    print(f"Entradas: {manifest['entry_count']}; dataset incluido: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
