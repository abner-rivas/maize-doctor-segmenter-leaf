"""Métricas de estabilidad de explicaciones LIME entre corridas con seeds distintas.

Compartido por `scripts/checks/lime_stability.py` (auditoría manual) y
`scripts/pipeline/inference_report.py` (reporte de inferencia puntual).
"""

import json
from pathlib import Path

import numpy as np


def reconstruct_mask_and_weight_map(
    json_path: Path, npy_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruye la máscara de superpíxeles positivos y el weight_map per-píxel.

    Se apoya en los artefactos que `render_visual_explanation` ya persiste junto al PNG,
    evitando re-ejecutar LIME.

    @param {Path} json_path Sidecar .json con los pesos por superpíxel.
    @param {Path} npy_path Sidecar .npy con el mapa de segmentos.
    @returns {tuple[np.ndarray, np.ndarray]} Máscara booleana de segmentos con peso
        positivo y mapa de pesos per-píxel.
    """
    metadata = json.loads(json_path.read_text())
    segments = np.load(npy_path)

    weight_map = np.zeros(segments.shape, dtype=float)
    positive_mask = np.zeros(segments.shape, dtype=bool)
    for feature in metadata["top_features"]:
        segment_id, weight = feature["segment_id"], feature["weight"]
        weight_map[segments == segment_id] = weight
        if weight > 0:
            positive_mask[segments == segment_id] = True

    return positive_mask, weight_map


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calcula la intersección sobre unión de dos máscaras booleanas.

    @param {np.ndarray} mask_a Primera máscara booleana.
    @param {np.ndarray} mask_b Segunda máscara booleana.
    @returns {float} IoU en [0, 1]; 1.0 cuando ambas máscaras están vacías.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 1.0


def weight_map_correlation(weight_map_a: np.ndarray, weight_map_b: np.ndarray) -> float:
    """Correlación de Pearson entre dos mapas de pesos per-píxel.

    @param {np.ndarray} weight_map_a Primer mapa de pesos.
    @param {np.ndarray} weight_map_b Segundo mapa de pesos.
    @returns {float} Coeficiente de Pearson, o 0.0 si algún mapa es constante.
    """
    flat_a, flat_b = weight_map_a.ravel(), weight_map_b.ravel()
    if flat_a.std() == 0 or flat_b.std() == 0:
        return 0.0
    return float(np.corrcoef(flat_a, flat_b)[0, 1])


def pairwise_stability(
    masks: list[np.ndarray], weight_maps: list[np.ndarray]
) -> list[dict[str, float | int]]:
    """Compara corridas consecutivas de LIME sobre la misma imagen.

    @param {list[np.ndarray]} masks Máscaras positivas, una por seed, en orden.
    @param {list[np.ndarray]} weight_maps Mapas de pesos, una por seed, en orden.
    @returns {list[dict]} Un registro por par consecutivo con seeds, IoU y correlación.
    """
    return [
        {
            "seed_a": i,
            "seed_b": i + 1,
            "iou": mask_iou(masks[i], masks[i + 1]),
            "correlation": weight_map_correlation(weight_maps[i], weight_maps[i + 1]),
        }
        for i in range(len(masks) - 1)
    ]
