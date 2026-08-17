# Entorno local del segmentador

## Requisitos

- Python 3.11 o superior;
- `make`;
- Node.js 20.19 o superior sólo para construir la documentación;
- GPU CUDA únicamente para entrenamiento o inferencia acelerada.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env
make install
```

`make install` instala el paquete editable con herramientas de desarrollo, soporte de
segmentación y Modal. El entrenamiento cloud mantiene constraints propios para no
reemplazar la instalación CUDA provista por la plataforma.

## Variables

- `PROJECT_DATA_ROOT`: raíz opcional de datos derivados; por defecto `data/`.
- `OUTPUT_ROOT`: raíz opcional de artefactos; por defecto `outputs/`.
- `DATASET_ROOT`: sólo se requiere para utilidades históricas del piloto cuyos
  manifiestos apuntan a imágenes fuente.

Los datos activos del segmentador se esperan por defecto en
`data/leaf_detection/detector_dataset`.

## Comprobación

```bash
make leaf-segmentation-status
make leaf-segmentation-verify-locks
make leaf-segmentation-verify-splits
python -m pytest
make lint
```

Consulte `make help` para los guards de entrenamiento y los flujos cloud/Modal.
