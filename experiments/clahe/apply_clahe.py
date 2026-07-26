"""Experimento aislado: aplica CLAHE a las imágenes de `experiments/clahe/input/`.

Deliberadamente desacoplado del pipeline principal: no toca `raw/`, `clean/` ni
`outputs/`, no lee `DATASET_ROOT` y no importa `src.*` fuera del loader de imagen.
Sirve para evaluar a ojo si la ecualización adaptativa ayuda en hojas de maíz con
iluminación irregular antes de decidir si vale la pena integrarla al pipeline.

CLAHE se aplica solo al canal L de LAB: ecualizar los tres canales RGB por separado
desplaza el color, y el color es señal diagnóstica en clorosis por deficiencia de
nutrientes (nitrogen/phosphorus/potassium_deficiency).

Uso:
    python experiments/clahe/apply_clahe.py
    python experiments/clahe/apply_clahe.py --limit 10 --clip-limit 3.0 --tile-grid 8
    python experiments/clahe/apply_clahe.py --input-dir <dir> --no-comparisons
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

EXPERIMENT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = EXPERIMENT_ROOT / "input"
OUTPUT_DIR = EXPERIMENT_ROOT / "output"
COMPARISONS_DIR = EXPERIMENT_ROOT / "comparisons"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_and_normalize_image(image_path: Path) -> Image.Image:
    """
    Carga una imagen de disco y aplica normalización de formato en caliente.

    Réplica local de `src.data.loader.load_and_normalize_image` para mantener el
    experimento ejecutable sin el paquete instalado, conservando las dos garantías
    que importan: orientación EXIF corregida y 3 canales RGB estrictos.

    @param {Path} image_path Ruta de la imagen a cargar.
    @returns {Image.Image} Imagen RGB con orientación corregida.
    """
    with Image.open(image_path) as raw:
        img = ImageOps.exif_transpose(raw)
        if img is raw:  # Pillow antiguos pueden devolver el mismo objeto lazy
            img = raw.copy()

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def apply_clahe(rgb: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """
    Aplica CLAHE sobre el canal de luminancia (L) del espacio LAB.

    @param {np.ndarray} rgb Imagen RGB uint8 de forma (H, W, 3).
    @param {float} clip_limit Umbral de recorte de contraste; valores altos amplifican ruido.
    @param {int} tile_grid Lado de la grilla de tiles (tile_grid x tile_grid).
    @returns {np.ndarray} Imagen RGB uint8 con luminancia ecualizada.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    equalized = clahe.apply(lightness)

    merged = cv2.merge((equalized, green_red, blue_yellow))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def render_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    output_path: Path,
    title: str,
    clip_limit: float,
    tile_grid: int,
) -> None:
    """
    Genera un panel 2x2: imágenes enfrentadas arriba, histogramas de luminancia abajo.

    @param {np.ndarray} original Imagen RGB uint8 sin procesar.
    @param {np.ndarray} processed Imagen RGB uint8 tras CLAHE.
    @param {Path} output_path Ruta del PNG a escribir.
    @param {str} title Nombre de la imagen, usado en el supertítulo.
    @param {float} clip_limit Valor usado, reportado en el supertítulo.
    @param {int} tile_grid Valor usado, reportado en el supertítulo.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), height_ratios=[3, 1])

    axes[0][0].imshow(original)
    axes[0][0].set_title("Original")
    axes[0][1].imshow(processed)
    axes[0][1].set_title(f"CLAHE (clip={clip_limit}, grid={tile_grid}x{tile_grid})")
    for ax in axes[0]:
        ax.axis("off")

    for ax, image, color, label in (
        (axes[1][0], original, "#6b7280", "Original"),
        (axes[1][1], processed, "#c2410c", "CLAHE"),
    ):
        lightness = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0]
        ax.hist(lightness.ravel(), bins=64, range=(0, 255), color=color)
        ax.set_title(f"Luminancia - {label}", fontsize=9)
        ax.set_yticks([])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def collect_images(input_dir: Path, limit: int | None) -> list[Path]:
    """
    Lista las imágenes del directorio de entrada en orden determinista.

    @param {Path} input_dir Directorio a escanear (no recursivo).
    @param {int|None} limit Máximo de imágenes a devolver; None para todas.
    @returns {list[Path]} Rutas ordenadas alfabéticamente.
    """
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit] if limit is not None else paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default=str(INPUT_DIR), help=f"Directorio de entrada (default: {INPUT_DIR})."
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Destino de las procesadas.")
    parser.add_argument(
        "--comparisons-dir", default=str(COMPARISONS_DIR), help="Destino de los paneles."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Procesa solo las primeras N imágenes."
    )
    parser.add_argument(
        "--clip-limit", type=float, default=2.0, help="clipLimit de CLAHE (default: 2.0)."
    )
    parser.add_argument(
        "--tile-grid", type=int, default=8, help="Lado de la grilla de tiles (default: 8)."
    )
    parser.add_argument(
        "--no-comparisons", action="store_true", help="Omite los paneles comparativos."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"El directorio de entrada no existe: {input_dir}")

    images = collect_images(input_dir, args.limit)
    if not images:
        raise SystemExit(
            f"No se encontraron imágenes en {input_dir}\n"
            f"Extensiones aceptadas: {', '.join(IMAGE_EXTENSIONS)}"
        )

    output_dir = Path(args.output_dir)
    comparisons_dir = Path(args.comparisons_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Procesando {len(images)} imagen(es) de {input_dir}")
    grid = f"({args.tile_grid}, {args.tile_grid})"
    print(f"CLAHE: clipLimit={args.clip_limit}, tileGridSize={grid}\n")

    failed = 0
    for image_path in images:
        try:
            original = np.array(load_and_normalize_image(image_path))
            processed = apply_clahe(original, args.clip_limit, args.tile_grid)

            destination = output_dir / f"{image_path.stem}_clahe.png"
            Image.fromarray(processed).save(destination)

            if not args.no_comparisons:
                render_comparison(
                    original,
                    processed,
                    comparisons_dir / f"{image_path.stem}_compare.png",
                    image_path.name,
                    args.clip_limit,
                    args.tile_grid,
                )
            print(f"  OK  {image_path.name} -> {destination.name}")
        except Exception as exc:
            failed += 1
            print(f"  ERR {image_path.name}: {exc}", file=sys.stderr)

    print(f"\nProcesadas: {len(images) - failed}/{len(images)}")
    print(f"Salida:      {output_dir}")
    if not args.no_comparisons:
        print(f"Comparación: {comparisons_dir}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
