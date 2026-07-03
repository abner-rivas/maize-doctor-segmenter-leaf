import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# None cuando falta la variable: Path("") resolvería al cwd, que "existe" y burla los guards.
_raw_dataset_root = os.getenv("DATASET_ROOT", "").strip()
DATASET_ROOT: Path | None = Path(_raw_dataset_root) if _raw_dataset_root else None


def get_dataset_root() -> Path:
    """Devuelve DATASET_ROOT validado; falla con mensaje claro si no está configurado."""
    if DATASET_ROOT is None:
        raise SystemExit(
            "DATASET_ROOT no está definido. Copia .env.example a .env y configúralo "
            "(ver LOCAL.md, sección 3)."
        )
    return DATASET_ROOT


def get_output_root() -> Path:
    """PROJECT_ROOT/outputs — raíz de artefactos generados por el pipeline (splits,
    resultados de entrenamiento, reports), separada de DATASET_ROOT (datos fuente)."""
    return PROJECT_ROOT / "outputs"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
