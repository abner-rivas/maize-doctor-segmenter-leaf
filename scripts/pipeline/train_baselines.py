import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.fastvit  # noqa: F401 - registra modelos
import src.models.baselines.ghostnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
import src.models.baselines.shufflenet  # noqa: F401 - registra modelos
from src.config import PROJECT_ROOT, get_dataset_root, get_output_root, set_global_seed
from src.data.dataset import CornDataset, build_weighted_sampler
from src.data.transforms import CornTransformFactory
from src.explainability.visual_report import explain_model_visual
from src.models.registry import MODEL_REGISTRY
from src.training.common import (
    build_run_dir,
    resolve_model_names,
    select_device,
    update_latest_pointer,
    worker_init_fn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def train_baseline(
    model_name: str,
    splits_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    target_size: tuple[int, int],
    lime_cfg: dict | None = None,
) -> None:
    logger.info(f"[{model_name}] Iniciando entrenamiento")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory
    )

    model = MODEL_REGISTRY.build(model_name, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    started_at = datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = build_run_dir(output_dir, model_name, run_id)

    best_val_f1 = -1.0
    best_ckpt_path = run_dir / "best.pth"

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_labels = _evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        logger.info(
            f"[{model_name}] Epoch {epoch}/{epochs} - "
            f"loss: {train_loss:.4f} | val macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)

    # Evaluación final sobre test con el mejor checkpoint
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_preds, test_labels = _evaluate(model, test_loader, device)

    target_names = [train_dataset.idx_to_class[i] for i in range(num_classes)]
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_acc = sum(p == gt for p, gt in zip(test_preds, test_labels)) / len(test_labels)
    report = classification_report(
        test_labels, test_preds, target_names=target_names, zero_division=0
    )

    metrics = {
        "model": model_name,
        "run_id": run_id,
        "trained_at": started_at.isoformat(timespec="seconds"),
        "epochs": epochs,
        "best_val_macro_f1": round(best_val_f1, 6),
        "test_accuracy": round(test_acc, 6),
        "test_macro_f1": round(test_f1, 6),
        "classification_report": report,
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info(f"[{model_name}] Métricas guardadas en {metrics_path}")
    logger.info(f"[{model_name}] test accuracy={test_acc:.4f} | test macro-F1={test_f1:.4f}")

    update_latest_pointer(output_dir, model_name, run_id)

    if lime_cfg is not None:
        test_df = pd.read_csv(splits_dir / "test.csv")
        explain_model_visual(
            model=model,
            model_name=model_name,
            test_df=test_df,
            dataset_root=get_dataset_root(),
            idx_to_class=train_dataset.idx_to_class,
            target_size=target_size,
            output_dir=run_dir,
            images_per_class=lime_cfg["images_per_class"],
            num_features=lime_cfg["num_features"],
            num_samples=lime_cfg["num_samples"],
            seed=lime_cfg["seed"],
            device=device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena modelos baseline de Deep Learning.")
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
        help="Directorio con train/val/test.csv. Si se omite, usa "
        "<repo>/outputs/splits/seed_42_baseline con --baseline, o "
        "<repo>/outputs/splits/seed_42 sin él.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Cuando no se pasa --splits-dir, usa el subset de splits/seed_42_baseline "
        "en vez del dataset completo (splits/seed_42).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio de salida para checkpoints y métricas "
        "(default: <repo>/outputs/baselines)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--lime",
        action="store_true",
        help="Tras entrenar cada modelo, genera el reporte visual LIME (config/dataset.yaml "
        "-> lime:) sobre una muestra balanceada del test set, igual que scripts/pipeline/"
        "explain_lime.py pero encadenado al propio entrenamiento.",
    )
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
    if args.splits_dir is not None:
        splits_dir = Path(args.splits_dir)
    elif args.baseline:
        splits_dir = output_root / "splits" / "seed_42_baseline"
    else:
        splits_dir = output_root / "splits" / "seed_42"
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "baselines"

    if not splits_dir.exists():
        raise SystemExit(
            f"El directorio de splits no existe: {splits_dir}\n"
            "Genera los splits primero con: make splits  (o make splits-baseline para el subset)"
        )

    device = select_device()
    target_size = tuple(cfg["dataset"]["target_size"])
    lime_cfg = cfg["lime"] if args.lime else None

    logger.info(f"Modelos a entrenar: {model_names}")
    logger.info(f"Splits: {splits_dir}  |  Epochs: {args.epochs}")

    for model_name in model_names:
        train_baseline(
            model_name=model_name,
            splits_dir=splits_dir,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            seed=seed,
            target_size=target_size,
            lime_cfg=lime_cfg,
        )

    logger.info("Entrenamiento de baselines completado.")


if __name__ == "__main__":
    main()
