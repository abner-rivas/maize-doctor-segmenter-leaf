import logging
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, WeightedRandomSampler

from src.config import PROJECT_ROOT, get_dataset_root
from src.data.loader import load_and_normalize_image
from src.data.transforms import MINORITY_CLASSES

_DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "dataset.yaml")

# Reintentos ante imágenes ilegibles antes de asumir que el dataset entero es inaccesible.
_MAX_FALLBACK_ATTEMPTS = 5

logger = logging.getLogger(__name__)


class CornDataset(Dataset):
    """
    Componente Dataset personalizado para el mapeo y consumo indexado
    de imágenes de patologías y deficiencias en hojas de maíz.
    """

    def __init__(
        self,
        csv_path: str,
        config_path: str = _DEFAULT_CONFIG,
        transform=None,
        minority_transform=None,
        exclude_classes: list[str] | None = None,
        class_to_idx: dict[str, int] | None = None,
    ):
        """
        Args:
            csv_path: Ruta al manifiesto del split (train.csv, val.csv o test.csv).
            config_path: Ruta al archivo de configuración paramétrica.
            transform: Pipeline de transformaciones estándar (torchvision).
            minority_transform: Pipeline extendido aplicado a clases en MINORITY_CLASSES.
                                Si None, todas las muestras usan `transform`.
            exclude_classes: Clases a excluir del dataset en tiempo de construcción.
                             El CSV permanece inmutable; la exclusión es una decisión de pipeline.
            class_to_idx: Mapeo canónico clase->índice a reutilizar (el del split de train,
                          inyectado en val/test) para mantener índices consistentes entre
                          splits. Si None, se construye desde el YAML.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontró el archivo de manifiesto: {csv_path}")

        self.transform = transform
        self.minority_transform = minority_transform
        self.dataset_root = get_dataset_root()

        # 1. Cargar y filtrar el manifiesto
        df = pd.read_csv(csv_path)
        if exclude_classes:
            df = df[~df["label"].isin(exclude_classes)].reset_index(drop=True)
        self.data_frame = df

        present = set(self.data_frame["label"].unique())

        if class_to_idx is not None:
            # Reutilizar el mapeo inyectado, validando cobertura total.
            unknown = present - set(class_to_idx)
            if unknown:
                raise ValueError(
                    f"Etiquetas en el CSV sin índice en el mapeo inyectado: {sorted(unknown)}"
                )
            self.class_to_idx = dict(class_to_idx)
            self.allowed_classes = sorted(self.class_to_idx, key=self.class_to_idx.__getitem__)
        else:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Construir class_to_idx compacto solo con las clases presentes tras el filtro.
            # El orden respeta la lista del YAML para reproducibilidad entre ejecuciones.
            self.allowed_classes = [c for c in config["dataset"]["classes"] if c in present]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.allowed_classes)}

            # Validar que no haya etiquetas en el CSV no cubiertas por el YAML.
            # Falla en construcción, no en el primer batch.
            unknown = present - set(config["dataset"]["classes"])
            if unknown:
                raise ValueError(f"Etiquetas en el CSV no registradas en config: {sorted(unknown)}")

        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def __len__(self) -> int:
        """Devuelve el tamaño neto total de la muestra actual."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Carga perezosa: lee, normaliza y transforma la muestra bajo demanda."""
        # Reintentos acotados: una imagen corrupta no mata el worker, pero una racha de
        # fallos (dataset inaccesible) sí se propaga.
        last_error: Exception | None = None
        for attempt in range(_MAX_FALLBACK_ATTEMPTS):
            row = self.data_frame.iloc[(idx + attempt) % len(self)]
            img_path = self.dataset_root / row["image_path"]
            try:
                image = load_and_normalize_image(img_path)
                class_name = row["label"]
                break
            except (FileNotFoundError, RuntimeError) as e:
                last_error = e
                logger.warning(
                    f"Imagen no disponible en idx={idx + attempt} ({img_path}): {e}. "
                    "Probando la siguiente fila."
                )
        else:
            raise RuntimeError(
                f"{_MAX_FALLBACK_ATTEMPTS} imágenes consecutivas ilegibles desde idx={idx}; "
                "verifica que DATASET_ROOT siga accesible y los splits estén al día."
            ) from last_error

        # 2. Mapear la etiqueta de texto a su correspondiente índice entero codificado
        label_idx = self.class_to_idx[class_name]

        # 3. Seleccionar pipeline: extendido para clases minoritarias, estándar para el resto
        pipeline = (
            self.minority_transform
            if self.minority_transform is not None and class_name in MINORITY_CLASSES
            else self.transform
        )
        if pipeline:
            image = pipeline(image)

        return image, label_idx


def build_weighted_sampler(dataset: "CornDataset", seed: int) -> WeightedRandomSampler:
    labels = dataset.data_frame["label"].tolist()
    class_counts = dataset.data_frame["label"].value_counts().to_dict()
    sample_weights = torch.tensor(
        [1.0 / class_counts[label] for label in labels], dtype=torch.float
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
