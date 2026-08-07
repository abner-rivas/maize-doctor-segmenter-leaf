import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from PIL import Image

from src.config import PROJECT_ROOT, get_output_root, get_project_data_root
from src.data.dataset import resolve_class_mapping
from src.data.loader import load_and_normalize_image
from src.data.transforms import CornTransformFactory
from src.inference.classifier import ClassificationPrediction, classify_image
from src.inference.dual_perspective import (
    DOMAIN_SHIFT_WARNING,
    DualPerspectiveConfig,
    DualPerspectiveResult,
    classify_dual_perspective,
)
from src.models import build_model, list_models
from src.preprocessing.segmented_leaf_processor import (
    BASELINE_FULL,
    SUPPORTED_MASK_PROFILES,
    SegmentedLeafProcessor,
    mask_processor_config_from_mapping,
)
from src.segmentation.leaf_segmenter import UltralyticsLeafSegmenter
from src.training.common import resolve_run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_SIZE_BY_MODEL = {
    "mobilenet_v3_large": 224,
    "mobilenet_v3_small": 224,
    "efficientnet_b4": 380,
}


def _load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _default_splits_dir(project_data_root: Path, baseline: bool, full: bool) -> Path:
    if baseline and full:
        raise SystemExit("Usa solo uno de --baseline o --full.")
    if baseline:
        return project_data_root / "splits" / "seed_42_baseline"
    if full:
        return project_data_root / "splits" / "seed_42"

    baseline_dir = project_data_root / "splits" / "seed_42_baseline"
    return baseline_dir if baseline_dir.exists() else project_data_root / "splits" / "seed_42"


def _load_summary(checkpoint_path: Path) -> dict[str, Any]:
    summary_path = checkpoint_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r") as f:
        return json.load(f)


def _resolve_class_mapping(
    summary: dict[str, Any],
    splits_dir: Path,
    cfg: dict[str, Any],
) -> tuple[dict[str, int], dict[int, str]]:
    if "class_to_idx" in summary:
        class_to_idx = {str(name): int(idx) for name, idx in summary["class_to_idx"].items()}
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        return class_to_idx, idx_to_class

    train_csv = splits_dir / "train.csv"
    if not train_csv.exists():
        raise SystemExit(
            f"No existe {train_csv}. Pasa --splits-dir o conserva summary.json junto al checkpoint."
        )
    return resolve_class_mapping(str(train_csv), cfg["dataset"]["classes"])


def _resolve_target_size(
    args: argparse.Namespace,
    summary: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[int, int]:
    if args.image_size is not None:
        return (args.image_size, args.image_size)

    image_size = summary.get("image_size")
    if isinstance(image_size, list) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))

    default_size = _DEFAULT_IMAGE_SIZE_BY_MODEL.get(args.model)
    if default_size is not None:
        return (default_size, default_size)

    height, width = cfg["dataset"]["target_size"]
    return (height, width)


def _load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Checkpoint invalido: {checkpoint_path}")
    return checkpoint


def _build_leaf_processor(
    *,
    cfg: dict[str, Any],
    output_root: Path,
    checkpoint_override: Path | None,
    segmenter_device: str | None,
    processing_profile: str,
    target_size: tuple[int, int],
) -> SegmentedLeafProcessor:
    leaf_cfg = cfg["leaf_detection"]
    segmentation_cfg = leaf_cfg["segmentation"]
    segmenter_checkpoint = checkpoint_override or (
        output_root / str(segmentation_cfg["checkpoint"])
    )
    segmenter = UltralyticsLeafSegmenter(
        segmenter_checkpoint,
        image_size=int(segmentation_cfg["image_size"]),
        confidence_threshold=float(leaf_cfg["confidence_threshold"]),
        iou_threshold=float(segmentation_cfg["iou_threshold"]),
        max_detections=int(segmentation_cfg["max_detections"]),
        device=segmenter_device,
        expected_version=str(segmentation_cfg["ultralytics_version"]),
    )
    return SegmentedLeafProcessor(
        segmenter,
        mask_processor_config_from_mapping(
            leaf_cfg,
            processing_profile=processing_profile,
            target_size=target_size,
        ),
    )


def _print_top_k(prediction: ClassificationPrediction) -> None:
    print("Top-k:")
    for ranked in prediction.top_k:
        print(f"  {ranked.class_name}: {ranked.probability:.4f}")


def _print_dual_result(result: DualPerspectiveResult) -> None:
    full = result.full_image
    segmented = result.segmented_leaf.prediction
    print(f"Imagen completa: {full.class_name} ({full.confidence:.4f})")
    if segmented is None:
        print(
            "Hoja segmentada: No disponible - "
            f"{result.segmented_leaf.reason or 'segmentación no confiable'}"
        )
    else:
        print(f"Hoja segmentada: {segmented.class_name} ({segmented.confidence:.4f})")
    print(f"Estado de segmentación: {result.segmentation.status.value}")
    print(f"Agreement: {result.agreement}")
    print("Resultado estructurado:")
    print(
        json.dumps(
            result.to_metadata(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice la clase de una imagen de hoja de maiz.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list_models(),
        help="Nombre interno del modelo registrado.",
    )
    parser.add_argument("--image", required=True, help="Ruta a la imagen a clasificar.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta al checkpoint .pth (default: último run en <repo>/outputs/baselines).",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="run_id específico a usar. Por defecto usa latest.json para el modelo.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train.csv para reconstruir class_to_idx si no hay summary.json.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Usa PROJECT_DATA_ROOT/splits/seed_42_baseline.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Usa PROJECT_DATA_ROOT/splits/seed_42.",
    )
    parser.add_argument("--image-size", type=int, default=None, dest="image_size")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k")
    parser.add_argument(
        "--leaf-profile",
        choices=sorted(SUPPORTED_MASK_PROFILES),
        default=BASELINE_FULL,
        help=(
            "Preprocesamiento opt-in. baseline_full conserva la entrada histórica; "
            "los demás perfiles introducen cambio de representación."
        ),
    )
    parser.add_argument("--segmenter-checkpoint", type=Path, default=None)
    parser.add_argument("--leaf-debug-dir", type=Path, default=None)
    parser.add_argument("--segmenter-device", default=None)
    parser.add_argument(
        "--dual-perspective",
        action="store_true",
        help=(
            "Ejecuta, de forma experimental y opt-in, clasificación independiente "
            "de imagen completa y hoja segmentada."
        ),
    )
    parser.add_argument(
        "--dual-output-json",
        type=Path,
        default=None,
        help="Escribe el resultado estructurado sin sobrescribir un archivo existente.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
    )
    args = parser.parse_args()
    if args.dual_perspective and args.leaf_profile != BASELINE_FULL:
        parser.error(
            "--dual-perspective no se combina con --leaf-profile; la política dual "
            "usa leaf_detection.dual_perspective.segmented_profile"
        )
    if args.dual_output_json is not None and not args.dual_perspective:
        parser.error("--dual-output-json requiere --dual-perspective")

    output_root = get_output_root()
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = (
            resolve_run_dir(
                output_root / "baselines",
                args.model,
                args.run,
            )
            / "best.pth"
        )
    if not checkpoint_path.exists():
        raise SystemExit(f"No existe el checkpoint: {checkpoint_path}")

    config_path = Path(args.config)
    cfg = _load_config(config_path)
    summary = _load_summary(checkpoint_path)
    splits_dir = (
        Path(args.splits_dir)
        if args.splits_dir
        else _default_splits_dir(
            get_project_data_root(),
            baseline=args.baseline,
            full=args.full,
        )
    )
    class_to_idx, idx_to_class = _resolve_class_mapping(summary, splits_dir, cfg)
    target_size = _resolve_target_size(args, summary, cfg)

    device = select_device()
    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(_load_state_dict(checkpoint_path, device))
    model.eval()

    factory = CornTransformFactory(config_path=str(config_path), target_size=target_size)
    inference_transform = factory.get_pipeline("inference")
    original_image = load_and_normalize_image(args.image)

    def classifier(candidate: Image.Image) -> ClassificationPrediction:
        return classify_image(
            model,
            candidate,
            transform=lambda current: cast(torch.Tensor, inference_transform(current)),
            idx_to_class=idx_to_class,
            device=device,
            top_k=args.top_k,
        )

    print(f"Modelo: {args.model}")
    print(f"Imagen: {args.image}")
    if args.dual_perspective:
        logger.warning(DOMAIN_SHIFT_WARNING)
        policy = DualPerspectiveConfig.from_mapping(cfg["leaf_detection"])
        leaf_processor = _build_leaf_processor(
            cfg=cfg,
            output_root=output_root,
            checkpoint_override=args.segmenter_checkpoint,
            segmenter_device=args.segmenter_device,
            processing_profile=policy.segmented_profile,
            target_size=target_size,
        )
        dual_result = classify_dual_perspective(
            original_image,
            classifier=classifier,
            leaf_processor=leaf_processor,
            config=policy,
            source_image=args.image,
            debug_dir=args.leaf_debug_dir,
        )
        _print_dual_result(dual_result)
        if args.dual_output_json is not None:
            if args.dual_output_json.exists():
                raise SystemExit(
                    f"No se sobrescribe el resultado existente: {args.dual_output_json}"
                )
            args.dual_output_json.parent.mkdir(parents=True, exist_ok=True)
            args.dual_output_json.write_text(
                json.dumps(
                    dual_result.to_metadata(),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        return

    image = original_image
    leaf_metadata: dict[str, object] | None = None
    if args.leaf_profile != BASELINE_FULL:
        logger.warning(
            "El checkpoint clasificador histórico fue entrenado con imágenes completas. "
            "El perfil %s es experimental y puede introducir domain shift.",
            args.leaf_profile,
        )
        leaf_processor = _build_leaf_processor(
            cfg=cfg,
            output_root=output_root,
            checkpoint_override=args.segmenter_checkpoint,
            segmenter_device=args.segmenter_device,
            processing_profile=args.leaf_profile,
            target_size=target_size,
        )
        leaf_result = leaf_processor.process(
            image,
            source_image=args.image,
            debug_dir=args.leaf_debug_dir,
        )
        if leaf_result.processed_image is None:
            raise SystemExit(
                f"El preprocesamiento rechazó la imagen: {leaf_result.fallback_reason}"
            )
        if leaf_result.fallback_used:
            logger.warning("Fallback de segmentación: %s", leaf_result.fallback_reason)
        image = leaf_result.processed_image
        leaf_metadata = leaf_result.to_metadata()
    prediction = classifier(image)
    if leaf_metadata is not None:
        print("Preprocesamiento de hoja:")
        print(json.dumps(leaf_metadata, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Prediccion: {prediction.class_name} ({prediction.confidence:.4f})")
    _print_top_k(prediction)


if __name__ == "__main__":
    main()
