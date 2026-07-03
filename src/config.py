import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# None (y no Path("")) cuando la variable falta: Path("") resuelve al cwd, que "existe",
# por lo que los guards `DATASET_ROOT.exists()` de los entrypoints pasaban con una
# configuración ausente (y p.ej. la descarga del dataset caía dentro del repo).
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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
