"""Utilidades compartidas por los scripts de entrenamiento (train.py y train_baselines.py)."""

import logging
import random

import numpy as np
import torch

from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


def resolve_model_names(requested: list[str], registry: ModelRegistry) -> list[str]:
    available = registry.list_names()
    if requested == ["all"]:
        return available
    unknown = [n for n in requested if n not in registry]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {available}")
    return requested


def worker_init_fn(worker_id: int) -> None:
    """Propaga la semilla per-worker de PyTorch (variable por época) a `random` y `numpy`."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device() -> torch.device:
    """Selecciona cuda si está disponible y deja registro del hardware detectado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Dispositivo: GPU - {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")
    else:
        logger.warning(
            "Dispositivo: CPU (no se detectó GPU — el entrenamiento será "
            "significativamente más lento)"
        )
    return device
