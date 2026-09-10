#!/usr/bin/env python3
# ruff: noqa: E501, I001
"""Generate deterministic figures for the Doctor Maiz segmenter report."""

from __future__ import annotations

import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("SEGMENTER_REPO", str(REPORT_DIR.parents[1]))
).expanduser().resolve()
RUN_DIR = (
    REPO_ROOT
    / "outputs/full_dataset_inference/2026-08-27_updated_hf_dataset"
)
MANIFEST = RUN_DIR / "manifest.csv"
FIGURES = REPORT_DIR / "figures"

GREEN = "#16803a"
GREEN_LIGHT = "#72c472"
YELLOW = "#e7a425"
RED = "#c83e3e"
BLUE = "#3478b8"
GRAY = "#667085"
LIGHT = "#edf5ef"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pct(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_values(rows: list[dict[str, str]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(field, "")
        if raw:
            try:
                values.append(float(raw))
            except ValueError:
                continue
    return np.asarray(values, dtype=float)


def dataset_splits() -> None:
    labels = ["Train", "Validacion", "Test"]
    images = np.asarray([809, 173, 173])
    masks = np.asarray([858, 183, 183])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    width = 0.34
    left = ax.bar(x - width / 2, images, width, label="Imagenes", color=GREEN)
    right = ax.bar(x + width / 2, masks, width, label="Mascaras", color=BLUE)
    ax.bar_label(left, padding=3)
    ax.bar_label(right, padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Conteo")
    ax.set_title("Splits congelados del dataset de segmentacion")
    ax.legend(frameon=False)
    ax.set_ylim(0, 960)
    save(fig, "dataset_splits.png")


def source_splits() -> None:
    labels = ["Train", "Validacion", "Test"]
    corn = np.asarray([109, 23, 23])
    disease = np.asarray([700, 150, 150])
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(labels, corn, color=YELLOW, label="corn")
    ax.bar(
        labels,
        disease,
        bottom=corn,
        color=GREEN,
        label="corn_leaf_diseases_classification",
    )
    for index, total in enumerate(corn + disease):
        ax.text(index, total + 13, str(total), ha="center", fontweight="bold")
    ax.set_ylabel("Imagenes")
    ax.set_title("Composicion por fuente en cada split")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 900)
    save(fig, "source_splits.png")


def model_comparison() -> None:
    names = ["C-01 baseline\nsemilla 42", "D-01 mosaic=0\nsemilla 42"]
    values = [0.93806, 0.94399]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    bars = ax.bar(names, values, color=[GRAY, GREEN], width=0.55)
    ax.bar_label(bars, labels=[f"{value:.5f}" for value in values], padding=4)
    ax.set_ylabel("Mask mAP50-95 (mejor epoca en val)")
    ax.set_ylim(0.90, 0.955)
    ax.set_title("Ablacion controlada: efecto de desactivar mosaic")
    ax.text(
        0.5,
        0.906,
        "+0.00593 (+0.59 puntos porcentuales)",
        ha="center",
        color=GREEN,
        fontweight="bold",
    )
    save(fig, "model_comparison.png")


def d01_metrics() -> None:
    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    masks = np.asarray([0.99424, 0.94309, 0.97326, 0.94404])
    boxes = np.asarray([1.00000, 0.94855, 0.97979, 0.94343])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    width = 0.36
    ax.bar(x - width / 2, masks, width, label="Mascara", color=GREEN)
    ax.bar(x + width / 2, boxes, width, label="Caja", color=BLUE)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.90, 1.008)
    ax.set_ylabel("Metrica en validacion")
    ax.set_title("Checkpoint D-01 en el split de validacion")
    ax.legend(frameon=False, ncol=2)
    for index, value in enumerate(masks):
        ax.text(index - width / 2, value + 0.002, f"{value:.3f}", ha="center", fontsize=8)
    for index, value in enumerate(boxes):
        ax.text(index + width / 2, value + 0.002, f"{value:.3f}", ha="center", fontsize=8)
    save(fig, "d01_metrics.png")


def downstream_metrics() -> None:
    labels = ["IoU", "Dice", "Recall de hoja", "Precision de hoja"]
    values = [0.98122, 0.99046, 0.99375, 0.98731]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(labels, values, color=[GREEN, GREEN_LIGHT, BLUE, YELLOW])
    ax.bar_label(bars, labels=[f"{value:.5f}" for value in values], padding=3)
    ax.set_ylim(0.86, 1.005)
    ax.set_ylabel("Media por imagen")
    ax.set_title("D-01: evaluacion end-to-end sobre 150 imagenes de val")
    ax.tick_params(axis="x", rotation=10)
    save(fig, "downstream_metrics.png")


def status_by_class(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row["class"]][row["reliability_status"]] += 1
    classes = sorted(grouped)
    short = [
        {
            "common_rust": "Roya comun",
            "fall_armyworm": "Cogollero",
            "gray_leaf_spot": "Mancha gris",
            "healthy": "Sana",
            "lethal_necrosis": "Necrosis letal",
            "nitrogen_deficiency": "Def. nitrogeno",
            "northern_corn_leaf_blight": "Tizon norteno",
            "phosphorus_deficiency": "Def. fosforo",
            "potassium_deficiency": "Def. potasio",
        }[name]
        for name in classes
    ]
    totals = np.asarray([sum(grouped[name].values()) for name in classes], dtype=float)
    reliable = np.asarray([grouped[name]["reliable"] for name in classes]) / totals * 100
    uncertain = np.asarray([grouped[name]["uncertain"] for name in classes]) / totals * 100
    failed = np.asarray([grouped[name]["failed"] for name in classes]) / totals * 100
    y = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.barh(y, reliable, color=GREEN, label="Reliable")
    ax.barh(y, uncertain, left=reliable, color=YELLOW, label="Uncertain")
    ax.barh(y, failed, left=reliable + uncertain, color=RED, label="Failed")
    ax.set_yticks(y, short)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Porcentaje de imagenes")
    ax.set_title("Resultado del quality gate por clase (N=33,437)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.19))
    for index, value in enumerate(reliable):
        ax.text(value - 1.0, index, f"{value:.1f}%", va="center", ha="right", color="white", fontsize=8)
    fig.subplots_adjust(bottom=0.20)
    save(fig, "status_by_class.png")


def rejection_reasons(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["rejection_reason"] for row in rows if row["rejection_reason"])
    labels = {
        "no_detection": "Sin deteccion",
        "low_segmentation_confidence": "Confianza baja",
        "invalid_mask": "Mascara invalida",
        "ambiguous_instance_score_margin": "Margen ambiguo",
        "excessive_mask_area_ratio": "Area excesiva",
        "suspicious_large_mask_geometry": "Geometria sospechosa",
    }
    ordered = sorted(counts, key=lambda name: counts[name])
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.barh(
        [labels[name] for name in ordered],
        [counts[name] for name in ordered],
        color=[RED if name in {"no_detection", "low_segmentation_confidence", "invalid_mask"} else YELLOW for name in ordered],
    )
    ax.bar_label(bars, padding=4)
    ax.set_xlabel("Imagenes")
    ax.set_title("Motivo primario de las 5,077 salidas no confiables")
    save(fig, "rejection_reasons.png")


def class_retention(rows: list[dict[str, str]]) -> None:
    totals = Counter(row["class"] for row in rows)
    kept = Counter(row["class"] for row in rows if row["reliability_status"] == "reliable")
    classes = sorted(totals)
    short = [
        {
            "common_rust": "Roya",
            "fall_armyworm": "Cogollero",
            "gray_leaf_spot": "Mancha gris",
            "healthy": "Sana",
            "lethal_necrosis": "Necrosis",
            "nitrogen_deficiency": "Nitrogeno",
            "northern_corn_leaf_blight": "Tizon",
            "phosphorus_deficiency": "Fosforo",
            "potassium_deficiency": "Potasio",
        }[name]
        for name in classes
    ]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    width = 0.37
    ax.bar(x - width / 2, [totals[c] for c in classes], width, color=GRAY, label="Entrada")
    ax.bar(x + width / 2, [kept[c] for c in classes], width, color=GREEN, label="Exportada")
    ax.set_xticks(x, short, rotation=25, ha="right")
    ax.set_ylabel("Imagenes")
    ax.set_title("Retencion para el dataset del clasificador")
    ax.legend(frameon=False)
    save(fig, "class_retention.png")


def mask_distribution(rows: list[dict[str, str]]) -> None:
    reliable = float_values(
        [row for row in rows if row["reliability_status"] == "reliable"],
        "mask_area_ratio",
    )
    uncertain = float_values(
        [row for row in rows if row["reliability_status"] == "uncertain"],
        "mask_area_ratio",
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bins = np.linspace(0, 1, 41)
    ax.hist(reliable, bins=bins, density=True, alpha=0.72, color=GREEN, label=f"Reliable (n={len(reliable):,})")
    ax.hist(uncertain, bins=bins, density=True, alpha=0.65, color=YELLOW, label=f"Uncertain (n={len(uncertain):,})")
    ax.axvline(0.71829, color=BLUE, linestyle="--", linewidth=1.6, label="Mediana global 0.718")
    ax.set_xlabel("Fraccion de la imagen cubierta por la mascara")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribucion del area de las 29,417 mascaras seleccionadas")
    ax.legend(frameon=False)
    save(fig, "mask_area_distribution.png")


def inference_distribution(rows: list[dict[str, str]]) -> None:
    values = float_values(rows, "inference_time_ms")
    clipped = values[values <= np.quantile(values, 0.99)]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.hist(clipped, bins=55, color=BLUE, alpha=0.82)
    ax.axvline(np.median(values), color=GREEN, linestyle="--", linewidth=1.7, label=f"Mediana {np.median(values):.2f} ms")
    ax.axvline(np.mean(values), color=RED, linestyle=":", linewidth=1.7, label=f"Media {np.mean(values):.2f} ms")
    ax.set_xlabel("Tiempo de inferencia por imagen (ms; recorte al p99)")
    ax.set_ylabel("Imagenes")
    ax.set_title("Latencia observada en CPU")
    ax.legend(frameon=False)
    save(fig, "inference_time_distribution.png")


def pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    stages = [
        (0.2, "Fuentes\nYOLO/COCO", GRAY),
        (2.15, "Auditoria y\nconsolidacion", BLUE),
        (4.1, "Splits sin fuga\n70/15/15", BLUE),
        (6.05, "YOLO26n-seg\nC-01 / D-01", GREEN),
        (8.0, "Seleccion de\ninstancia", GREEN),
        (9.95, "Quality gate y\nsalida", YELLOW),
    ]
    for index, (x, label, color) in enumerate(stages):
        box = plt.Rectangle((x, 0.9), 1.6, 1.15, facecolor=color, alpha=0.9, edgecolor="white", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.8, 1.475, label, ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if index + 1 < len(stages):
            ax.annotate("", xy=(x + 1.93, 1.475), xytext=(x + 1.62, 1.475), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#344054"})
    ax.text(10.75, 0.53, "reliable -> mascara negra", ha="center", color=GREEN, fontsize=8.5)
    ax.text(10.75, 0.24, "uncertain/failed -> original", ha="center", color=RED, fontsize=8.5)
    ax.set_title("Flujo tecnico del segmentador", pad=4, fontweight="bold")
    save(fig, "pipeline_architecture.png")


def qualitative_panel(rows: list[dict[str, str]]) -> None:
    ids = [
        "image_2e1dc6030ab674a66e7f",
        "image_060f65b6fbd56d9e1b47",
        "image_0eaa3a345425c74b5b78",
    ]
    by_id = {row["image_id"]: row for row in rows}
    columns = [
        ("original_thumbnail_path", "Original"),
        ("prediction_path", "Propuestas"),
        ("mask_path", "Mascara"),
        ("final_thumbnail_path", "Salida final"),
    ]
    fig, axes = plt.subplots(len(ids), len(columns), figsize=(11.0, 8.1))
    status_titles = {
        "reliable": "RELIABLE: healthy",
        "uncertain": "UNCERTAIN: geometria sospechosa",
        "failed": "FAILED: sin deteccion",
    }
    status_colors = {"reliable": GREEN, "uncertain": YELLOW, "failed": RED}
    for row_index, image_id in enumerate(ids):
        row = by_id[image_id]
        status = row["reliability_status"]
        for column_index, (field, title) in enumerate(columns):
            ax = axes[row_index, column_index]
            relative = row.get(field, "")
            path = RUN_DIR / relative if relative else None
            if path is not None and path.exists():
                with Image.open(path) as image:
                    shown = image.convert("RGB") if image.mode != "L" else image.copy()
                    ax.imshow(shown, cmap="gray" if image.mode == "L" else None)
            else:
                ax.set_facecolor("#f2f4f7")
                ax.text(0.5, 0.5, "No disponible", ha="center", va="center", transform=ax.transAxes, color=GRAY)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title, fontsize=10)
            if column_index == 0:
                ax.set_ylabel(status_titles[status], color=status_colors[status], fontweight="bold", fontsize=9)
    fig.suptitle("Casos deterministas del pipeline completo", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "qualitative_cases.png")


def copy_static_assets() -> None:
    assets = {
        REPO_ROOT / "public/logo.png": FIGURES / "logo.png",
        REPO_ROOT / "public/leaf_detection/external_sources_eda/inventory_counts.png": FIGURES / "eda_inventory.png",
        REPO_ROOT / "public/leaf_detection/external_sources_eda/polygon_area_by_class.png": FIGURES / "eda_polygon_area.png",
        REPO_ROOT / "public/leaf_detection/external_sources_eda/border_touching_percent.png": FIGURES / "eda_border_touching.png",
    }
    for source, destination in assets.items():
        if not source.is_file():
            raise FileNotFoundError(f"No existe el activo requerido: {source}")
        shutil.copyfile(source, destination)


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"No existe el manifiesto de inferencia: {MANIFEST}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    rows = read_rows()
    if len(rows) != 33_437:
        raise RuntimeError(f"Se esperaban 33437 filas; recibidas={len(rows)}")
    dataset_splits()
    source_splits()
    model_comparison()
    d01_metrics()
    downstream_metrics()
    status_by_class(rows)
    rejection_reasons(rows)
    class_retention(rows)
    mask_distribution(rows)
    inference_distribution(rows)
    pipeline_diagram()
    qualitative_panel(rows)
    copy_static_assets()
    print(f"Generated 12 analytical figures and 4 copied assets in {FIGURES}")


if __name__ == "__main__":
    main()
