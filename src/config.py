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

_raw_output_root = os.getenv("OUTPUT_ROOT", "").strip()
_raw_project_data_root = os.getenv("PROJECT_DATA_ROOT", "").strip()


def get_dataset_root() -> Path:
    """Devuelve DATASET_ROOT validado; falla con mensaje claro si no está configurado."""
    if DATASET_ROOT is None:
        raise SystemExit(
            "DATASET_ROOT no está definido. Copia .env.example a .env y configúralo "
            "(ver LOCAL.md, sección 3)."
        )
    return DATASET_ROOT


def get_output_root() -> Path:
    """OUTPUT_ROOT si está definido; si no, PROJECT_ROOT/outputs (default local).

    Env-overridable (mismo patrón que DATASET_ROOT) para redirigir artefactos a un
    volumen persistente en entornos remotos (p.ej. Modal). Sin OUTPUT_ROOT el
    comportamiento local no cambia."""
    return Path(_raw_output_root) if _raw_output_root else PROJECT_ROOT / "outputs"


def get_project_data_root() -> Path:
    """Return the root for reproducible data derived inside this project.

    ``DATASET_ROOT`` remains the large external source dataset. ``OUTPUT_ROOT``
    remains reserved for models, metrics, reports, previews, and diagnostics.
    """
    if _raw_project_data_root:
        return Path(_raw_project_data_root)
    return PROJECT_ROOT / "data"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
