#!/usr/bin/env python3
"""Register supplied yolo26s weights without downloading or replacing them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.config import get_output_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            get_output_root()
            / "leaf_detection"
            / "cloud_preflight"
            / "weights_manifest_yolo26s.json"
        ),
    )
    args = parser.parse_args()
    weights = args.weights.expanduser().resolve()
    if weights.name != "yolo26s-seg.pt" or not weights.is_file():
        raise SystemExit("Se requiere un archivo existente llamado yolo26s-seg.pt")
    payload = {
        "model": "yolo26s-seg",
        "path": str(weights),
        "sha256": sha256(weights),
        "size_bytes": weights.stat().st_size,
        "downloaded_by_script": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise SystemExit(f"No se reemplaza un manifiesto existente: {args.output}") from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
