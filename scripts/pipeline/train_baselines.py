import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, get_dataset_root, get_output_root, set_global_seed
from src.data.dataset import CornDataset, build_weighted_sampler
from src.data.transforms import CornTransformFactory
from src.explainability.augmentation_preview import save_augmentation_evidence
from src.models import MODEL_REGISTRY, build_model, list_models, resolve_input_size
from src.training.common import (
    build_run_dir,
    generate_run_id,
    resolve_model_names,
    select_device,
    update_latest_pointer,
    worker_init_fn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _base_target_size(cfg: dict) -> tuple[int, int]:
    height, width = cfg["dataset"]["target_size"]
    return (height, width)


def _resolve_model_target_size(
    model_name: str, args: argparse.Namespace, cfg: dict
) -> tuple[int, int]:
    """Resolución efectiva por modelo.
    """
    if args.image_size is not None:
        return (args.image_size, args.image_size)
    return resolve_input_size(model_name, _base_target_size(cfg))


def _scale_batch_size(base_batch: int, target_size: tuple[int, int], base: tuple[int, int]) -> int:
    """Escala el batch inversamente al área de la imagen para acotar la memoria de activaciones.
    """
    base_h, base_w = base
    height, width = target_size
    scaled = round(base_batch * (base_h * base_w) / (height * width))
    return max(1, min(base_batch, scaled))


def _build_dataloaders(
    splits_dir: Path,
    config_path: Path,
    target_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[
    DataLoader, DataLoader, DataLoader, dict[str, int], dict[int, str], CornTransformFactory
]:
    factory = CornTransformFactory(config_path=str(config_path), target_size=target_size)

    train_dataset = CornDataset(
        csv_path=str(splits_dir / "train.csv"),
        config_path=str(config_path),
        transform=factory.get_pipeline("train"),
        minority_transform=factory.get_pipeline("minority"),
    )
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = train_dataset.idx_to_class

    val_dataset = CornDataset(
        csv_path=str(splits_dir / "val.csv"),
        config_path=str(config_path),
        transform=factory.get_pipeline("val"),
        class_to_idx=class_to_idx,
    )
    test_dataset = CornDataset(
        csv_path=str(splits_dir / "test.csv"),
        config_path=str(config_path),
        transform=factory.get_pipeline("test"),
        class_to_idx=class_to_idx,
    )

    sampler = build_weighted_sampler(train_dataset, seed=seed)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class, factory


def _metrics_from_predictions(
    labels: list[int],
    predictions: list[int],
    loss: float,
) -> dict[str, float]:
    return {
        "loss": loss,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> tuple[dict[str, float], list[int], list[int], list[float]]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    seen = 0
    labels_all: list[int] = []
    preds_all: list[int] = []
    probs_all: list[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            labels_all.extend(labels.detach().cpu().tolist())
            probs = logits.detach().softmax(dim=1)
            preds_all.extend(probs.argmax(dim=1).cpu().tolist())
            probs_all.extend(probs.max(dim=1).values.cpu().tolist())

    avg_loss = running_loss / max(seen, 1)
    metrics = _metrics_from_predictions(labels_all, preds_all, avg_loss)
    return metrics, labels_all, preds_all, probs_all


def _write_test_outputs(
    model_dir: Path,
    idx_to_class: dict[int, str],
    labels: list[int],
    predictions: list[int],
) -> None:
    target_ids = sorted(idx_to_class)
    target_names = [idx_to_class[idx] for idx in target_ids]

    report = classification_report(
        labels,
        predictions,
        labels=target_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(model_dir / "test_classification_report.csv")

    matrix = confusion_matrix(labels, predictions, labels=target_ids)
    pd.DataFrame(matrix, index=target_names, columns=target_names).to_csv(
        model_dir / "test_confusion_matrix.csv"
    )


def _write_summary(
    run_dir: Path,
    model_name: str,
    run_id: str,
    args: argparse.Namespace,
    target_size: tuple[int, int],
    batch_size: int,
    class_to_idx: dict[str, int],
    splits_dir: Path,
    best_epoch: int,
    best_val_macro_f1: float,
    test_metrics: dict[str, float],
) -> None:
    summary = {
        "model": model_name,
        "run_id": run_id,
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "image_size": list(target_size),
        "splits_dir": str(splits_dir),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "pretrained": not args.no_pretrained,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test": test_metrics,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _train_model(
    model_name: str,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    args: argparse.Namespace,
    target_size: tuple[int, int],
    batch_size: int,
    splits_dir: Path,
    output_dir: Path,
    device: torch.device,
    factory: CornTransformFactory | None = None,
    seed: int = 42,
) -> Path:
    train_loader, val_loader, test_loader = loaders
    run_id = generate_run_id()
    run_dir = build_run_dir(output_dir, model_name, run_id)

    if factory is not None:
        save_augmentation_evidence(
            train_csv_path=str(splits_dir / "train.csv"),
            dataset_root=get_dataset_root(),
            train_transform=factory.get_pipeline("train"),
            minority_transform=factory.get_pipeline("minority"),
            output_dir=run_dir,
            minority_classes=train_loader.dataset.minority_classes,
            seed=seed,
        )

    logger.info("[%s] Construyendo modelo", model_name)
    model = build_model(
        model_name,
        num_classes=len(class_to_idx),
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float | int | str]] = []
    best_epoch = 0
    best_val_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        train_metrics, _, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            desc=f"{model_name} train {epoch}/{args.epochs}",
        )
        val_metrics, _, _, _ = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            desc=f"{model_name} val {epoch}/{args.epochs}",
        )
        epoch_seconds = perf_counter() - started

        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_epoch = epoch
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), run_dir / "best.pth")

        torch.save(model.state_dict(), run_dir / "last.pth")
        logger.info(
            "[%s] epoch %s/%s train_f1=%.4f val_f1=%.4f",
            model_name,
            epoch,
            args.epochs,
            train_metrics["macro_f1"],
            val_metrics["macro_f1"],
        )

    best_path = run_dir / "best.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics, labels, predictions, test_probs = _run_epoch(
        model,
        test_loader,
        criterion,
        device,
        desc=f"{model_name} test",
    )
    _write_test_outputs(run_dir, idx_to_class, labels, predictions)

    test_dataset = test_loader.dataset
    predictions_df = pd.DataFrame(
        {
            "image_path": test_dataset.data_frame["image_path"].tolist(),
            "label": test_dataset.data_frame["label"].tolist(),
            "pred_label": [idx_to_class[p] for p in predictions],
            "pred_prob": test_probs,
        }
    )
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)
    logger.info("[%s] Predicciones de test guardadas en %s", model_name, run_dir / "predictions.csv")

    _write_summary(
        run_dir,
        model_name,
        run_id,
        args,
        target_size,
        batch_size,
        class_to_idx,
        splits_dir,
        best_epoch,
        best_val_macro_f1,
        test_metrics,
    )
    update_latest_pointer(output_dir, model_name, run_id)
    logger.info("[%s] Test macro_f1=%.4f", model_name, test_metrics["macro_f1"])
    logger.info("[%s] Run completado en %s", model_name, run_dir)
    return run_dir


def _generate_lime_reports(
    model_name: str,
    run_dir: Path,
    splits_dir: Path,
    idx_to_class: dict[int, str],
    target_size: tuple[int, int],
    cfg: dict,
    device: torch.device,
    gradcam_enabled: bool = True,
) -> None:
    try:
        from src.explainability.visual_report import explain_model_visual
    except ModuleNotFoundError as e:
        logger.warning(
            "[%s] No se generaron reportes LIME porque falta la dependencia opcional: %s. "
            "Instala el extra xai con: pip install -e .[xai]",
            model_name,
            e.name,
        )
        return

    checkpoint_path = run_dir / "best.pth"
    if not checkpoint_path.exists():
        logger.warning("[%s] No se encontró best.pth para LIME en %s", model_name, run_dir)
        return

    lime_cfg = cfg["lime"]
    test_df = pd.read_csv(splits_dir / "test.csv")
    model = build_model(model_name, num_classes=len(idx_to_class), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    explain_model_visual(
        model=model,
        model_name=model_name,
        test_df=test_df,
        dataset_root=get_dataset_root(),
        idx_to_class=idx_to_class,
        target_size=target_size,
        output_dir=run_dir,
        images_per_class=lime_cfg["images_per_class"],
        num_features=lime_cfg["num_features"],
        num_samples=lime_cfg["num_samples"],
        seed=lime_cfg["seed"],
        device=device,
        enable_gradcam=gradcam_enabled,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena baselines de clasificacion.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f'Nombres de modelos, o "all". Disponibles: {list_models()}',
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Usa splits/seed_42_baseline en vez de splits/seed_42.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train/val/test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio base de salida (default: <repo>/outputs/baselines).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--image-size", type=int, default=None, dest="image_size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="learning_rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    parser.add_argument("--no-pretrained", action="store_true", dest="no_pretrained")
    parser.add_argument(
        "--lime",
        action="store_true",
        help="Genera reportes visuales LIME dentro del run al terminar cada modelo.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs debe ser mayor o igual a 1.")

    config_path = Path(args.config)
    cfg = _load_config(config_path)
    seed = cfg["baseline"]["seed"] if args.baseline else cfg["dataset"]["seed"]
    set_global_seed(seed)

    model_names = resolve_model_names(args.models, MODEL_REGISTRY)
    output_root = get_output_root()
    split_name = "seed_42_baseline" if args.baseline else "seed_42"
    splits_dir = Path(args.splits_dir) if args.splits_dir else output_root / "splits" / split_name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else output_root / "baselines"
    )
    base_target_size = _base_target_size(cfg)

    if not splits_dir.exists():
        command = "make splits-baseline" if args.baseline else "make splits"
        raise SystemExit(f"No existe {splits_dir}. Genera los splits primero con: {command}")

    device = select_device()
    gradcam_enabled = cfg.get("gradcam", {}).get("enabled", False)
    logger.info("Modelos a entrenar: %s", model_names)
    logger.info("Splits: %s", splits_dir)

    loader_cache: dict[tuple[tuple[int, int], int], tuple] = {}
    run_dirs: dict[str, Path] = {}
    for model_name in model_names:
        target_size = _resolve_model_target_size(model_name, args, cfg)
        batch_size = _scale_batch_size(args.batch_size, target_size, base=base_target_size)

        scaled = batch_size != args.batch_size or target_size != base_target_size
        logger.info(
            "[%s] tamano %dx%d, batch %d%s",
            model_name,
            target_size[0],
            target_size[1],
            batch_size,
            " (auto-escalado)" if scaled else "",
        )

        cache_key = (target_size, batch_size)
        if cache_key not in loader_cache:
            loader_cache[cache_key] = _build_dataloaders(
                splits_dir=splits_dir,
                config_path=config_path,
                target_size=target_size,
                batch_size=batch_size,
                num_workers=args.num_workers,
                seed=seed,
                device=device,
            )
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class, factory = loader_cache[
            cache_key
        ]

        run_dir = _train_model(
            model_name=model_name,
            loaders=(train_loader, val_loader, test_loader),
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            args=args,
            target_size=target_size,
            batch_size=batch_size,
            splits_dir=splits_dir,
            output_dir=output_dir,
            device=device,
            factory=factory,
            seed=seed,
        )
        run_dirs[model_name] = run_dir

        if args.lime:
            _generate_lime_reports(
                model_name=model_name,
                run_dir=run_dir,
                splits_dir=splits_dir,
                idx_to_class=idx_to_class,
                target_size=target_size,
                cfg=cfg,
                device=device,
                gradcam_enabled=gradcam_enabled,
            )

    if args.lime:
        logger.info("Reportes LIME procesados para runs: %s", run_dirs)


if __name__ == "__main__":
    main()
