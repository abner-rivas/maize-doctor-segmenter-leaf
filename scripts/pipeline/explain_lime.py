import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.fastvit  # noqa: F401 - registra modelos
import src.models.baselines.ghostnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
import src.models.baselines.shufflenet  # noqa: F401 - registra modelos
from src.config import DATASET_ROOT, PROJECT_ROOT, set_global_seed
from src.data.dataset import resolve_class_mapping
from src.data.loader import load_and_normalize_image
from src.explainability.visual_report import explain_model_visual, render_visual_explanation
from src.models.registry import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_RESULTS_DIR = DATASET_ROOT / "results" / "baselines"


def _resolve_model_names(requested: list[str]) -> list[str]:
    available = MODEL_REGISTRY.list_names()
    if requested == ["all"]:
        return available
    unknown = [n for n in requested if n not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {available}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera explicaciones LIME para checkpoints de baselines ya entrenados: "
        "por defecto, un reporte visual de 3 paneles por cada imagen de una muestra "
        "balanceada del test set (--images-per-class en config/dataset.yaml -> lime:); "
        "con --image, genera un único reporte puntual para esa imagen en vez del muestreo."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help='Nombres de modelos a explicar, o "all" para todos. '
        f"Disponibles: {MODEL_REGISTRY.list_names()}",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Ruta a una imagen puntual a explicar en vez del muestreo balanceado del "
        "test set. Se genera un reporte por cada modelo de --models.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta del PNG de salida. Solo válido junto con --image y un único modelo "
        "en --models. Default: <output_dir>/<model>/lime_visual/<stem-de-la-imagen>.png",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=None,
        help="Fuerza el uso de splits/seed_42_baseline en vez de leer lime.baseline de "
        "config/dataset.yaml.",
    )
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    lime_cfg = cfg["lime"]
    set_global_seed(lime_cfg["seed"])

    model_names = _resolve_model_names(args.models)
    use_baseline = args.baseline if args.baseline is not None else lime_cfg["baseline"]
    splits_dir = DATASET_ROOT / "splits" / ("seed_42_baseline" if use_baseline else "seed_42")
    classes = cfg["baseline"]["classes"] if use_baseline else cfg["dataset"]["classes"]

    if args.output is not None and (args.image is None or len(model_names) != 1):
        raise SystemExit("--output solo es válido junto con --image y un único modelo en --models.")

    if not splits_dir.exists():
        raise SystemExit(
            f"El directorio de splits no existe: {splits_dir}\n"
            "Genera los splits primero con: make splits  (o make splits-baseline)"
        )

    test_df = pd.read_csv(splits_dir / "test.csv") if args.image is None else None
    class_to_idx, idx_to_class = resolve_class_mapping(splits_dir / "train.csv", classes)
    num_classes = len(class_to_idx)
    target_size = tuple(cfg["dataset"]["target_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Modelos a explicar: {model_names}")

    for model_name in model_names:
        checkpoint_path = _RESULTS_DIR / model_name / "best.pth"
        if not checkpoint_path.exists():
            logger.warning(
                f"[{model_name}] No se encontró checkpoint en {checkpoint_path}, se omite. "
                "Entrena primero con: make train-baselines"
            )
            continue

        model = MODEL_REGISTRY.build(model_name, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        if args.image is not None:
            image_path = Path(args.image)
            output_path = (
                Path(args.output)
                if args.output is not None
                else _RESULTS_DIR / model_name / "lime_visual" / f"{image_path.stem}.png"
            )
            result = render_visual_explanation(
                image=load_and_normalize_image(image_path),
                model=model,
                idx_to_class=idx_to_class,
                target_size=target_size,
                output_path=output_path,
                num_samples=lime_cfg["num_samples"],
                num_features=lime_cfg["num_features"],
                seed=lime_cfg["seed"],
                device=device,
            )
            logger.info(
                f"[{model_name}] Diagnóstico: {result['predicted_label']} "
                f"(confianza: {result['predicted_prob'] * 100:.1f}%)"
            )
        else:
            explain_model_visual(
                model=model,
                model_name=model_name,
                test_df=test_df,
                dataset_root=DATASET_ROOT,
                idx_to_class=idx_to_class,
                target_size=target_size,
                output_dir=_RESULTS_DIR,
                images_per_class=lime_cfg["images_per_class"],
                num_features=lime_cfg["num_features"],
                num_samples=lime_cfg["num_samples"],
                seed=lime_cfg["seed"],
                device=device,
            )

    logger.info("Explicaciones LIME completadas.")


if __name__ == "__main__":
    main()
