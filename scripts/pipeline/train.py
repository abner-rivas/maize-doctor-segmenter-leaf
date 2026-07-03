import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
from src.config import PROJECT_ROOT, get_output_root, set_global_seed
from src.data.dataset import CornDataset, build_weighted_sampler
from src.data.transforms import CornTransformFactory
from src.models.registry import MODEL_REGISTRY
from src.training.common import (
    build_run_dir,
    generate_run_id,
    resolve_model_names,
    select_device,
    worker_init_fn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el pipeline principal de Deep Learning.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help='Nombres de modelos a entrenar, o "all" para todos. '
        f"Disponibles: {MODEL_REGISTRY.list_names()}",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train/val/test.csv (default: <repo>/outputs/splits/seed_42)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio de salida para checkpoints y métricas "
        "(default: <repo>/outputs/main)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["dataset"]["seed"]
    set_global_seed(seed)

    model_names = resolve_model_names(args.models, MODEL_REGISTRY)
    output_root = get_output_root()
    splits_dir = (
        Path(args.splits_dir) if args.splits_dir else output_root / "splits" / "seed_42"
    )
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "main"

    if not splits_dir.exists():
        raise SystemExit(
            f"El directorio de splits no existe: {splits_dir}\n"
            "Genera los splits primero con: make splits"
        )

    device = select_device()
    logger.info(f"Modelos a entrenar: {model_names}")

    factory = CornTransformFactory()
    train_dataset = CornDataset(
        csv_path=str(splits_dir / "train.csv"),
        transform=factory.get_pipeline("train"),
        minority_transform=factory.get_pipeline("minority"),
    )
    # Mapeo canónico del split de train, inyectado en val/test (índices consistentes)
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_to_idx)
    val_dataset = CornDataset(
        csv_path=str(splits_dir / "val.csv"),
        transform=factory.get_pipeline("val"),
        class_to_idx=class_to_idx,
    )
    test_dataset = CornDataset(
        csv_path=str(splits_dir / "test.csv"),
        transform=factory.get_pipeline("test"),
        class_to_idx=class_to_idx,
    )

    sampler = build_weighted_sampler(train_dataset, seed=seed)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(  # noqa: F841
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(  # noqa: F841
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(  # noqa: F841
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    # El desbalance ya lo compensa el sampler; no ponderar además la loss (ver CLAUDE.md)
    criterion = torch.nn.CrossEntropyLoss()  # noqa: F841

    for model_name in model_names:
        model = MODEL_REGISTRY.build(model_name, num_classes=num_classes).to(device)  # noqa: F841
        run_id = generate_run_id()
        run_dir = build_run_dir(output_dir, model_name, run_id)
        logger.info(f"[{model_name}] Modelo construido. Checkpoints en {run_dir}")

        # TODO: loop de entrenamiento — al implementarlo, guardar checkpoints/metrics en
        # run_dir y llamar a update_latest_pointer(output_dir, model_name, run_id) al final.


if __name__ == "__main__":
    main()
