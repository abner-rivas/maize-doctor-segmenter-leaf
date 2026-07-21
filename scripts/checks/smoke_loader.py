"""Read-only smoke check for datasets, transforms, loaders, sampler, and model registry.

This command never starts an optimizer, backward pass, epoch, or checkpoint write.
Models requested with ``--models`` are only constructed with ``pretrained=False``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, get_output_root
from src.data.dataset import CornDataset, build_weighted_sampler, resolve_class_mapping
from src.data.transforms import CornTransformFactory
from src.models import MODEL_REGISTRY, build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "dataset.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--check-sampler", action="store_true")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def _load_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError("Configuración inválida: falta dataset")
    return config


def _consume_batches(loader: DataLoader, limit: int, split: str) -> None:
    seen = 0
    for images, labels in loader:
        if images.ndim != 4 or labels.ndim != 1 or images.shape[0] != labels.shape[0]:
            raise RuntimeError(f"Batch inválido en {split}: {images.shape}, {labels.shape}")
        if not torch.isfinite(images).all():
            raise RuntimeError(f"Tensor no finito en {split}")
        seen += 1
        print(f"{split}: batch {seen} {tuple(images.shape)}")
        if seen >= limit:
            break
    if seen == 0:
        raise RuntimeError(f"Split vacío: {split}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.batches < 1:
        raise SystemExit("--batch-size y --batches deben ser positivos")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA solicitado pero no está disponible")

    config = _load_config(args.config)
    dataset_config = config["dataset"]
    classes = dataset_config.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("dataset.classes debe ser una lista no vacía")
    seed = int(config.get("baseline", {}).get("seed", dataset_config.get("seed", 42)))
    splits_dir = args.splits_dir or get_output_root() / "splits" / "seed_42_baseline"
    missing = [
        name for name in ("train", "val", "test") if not (splits_dir / f"{name}.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Faltan splits en {splits_dir}: {missing}")

    class_to_idx, _ = resolve_class_mapping(str(splits_dir / "train.csv"), classes)
    factory = CornTransformFactory(str(args.config))
    datasets = {
        split: CornDataset(
            csv_path=str(splits_dir / f"{split}.csv"),
            config_path=str(args.config),
            transform=factory.get_pipeline(split),
            class_to_idx=class_to_idx,
        )
        for split in ("train", "val", "test")
    }

    for split, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=min(args.batch_size, len(dataset)),
            shuffle=False,
            num_workers=0,
        )
        _consume_batches(loader, args.batches, split)

    if args.check_sampler:
        sampler = build_weighted_sampler(datasets["train"], seed=seed)
        if sampler is None:
            print("sampler: no requerido por la distribución configurada")
        else:
            first_indices = []
            for index, sampled in enumerate(sampler):
                first_indices.append(int(sampled))
                if index >= min(15, len(datasets["train"]) - 1):
                    break
            if not first_indices or any(index >= len(datasets["train"]) for index in first_indices):
                raise RuntimeError("WeightedRandomSampler produjo índices inválidos")
            print(f"sampler: OK ({len(first_indices)} índices inspeccionados)")

    unknown = sorted(set(args.models) - set(MODEL_REGISTRY.list_names()))
    if unknown:
        raise ValueError(f"Modelos no registrados: {unknown}")
    device = torch.device(args.device)
    for model_name in args.models:
        model = build_model(model_name, num_classes=len(class_to_idx), pretrained=False).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"modelo {model_name}: construido sin pesos preentrenados "
            f"({parameter_count} parámetros)"
        )
        del model

    print("Smoke check OK: no se ejecutaron forward, backward, optimizador, épocas ni checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
