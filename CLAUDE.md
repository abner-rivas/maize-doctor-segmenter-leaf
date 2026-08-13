# CLAUDE.md - DoctorMaiz Leaf Segmentation

## Alcance

Este repositorio contiene exclusivamente el proyecto del segmentador de hojas. No
añadir modelos, datasets, entrenamiento, inferencia ni explicabilidad del clasificador.

## Datos

- Las fuentes y datos derivados del segmentador viven bajo `data/leaf_detection/`.
- No modificar fuentes congeladas ni locks sin ejecutar el pipeline reproducible que
  los genera.
- `data/` contiene entradas y manifiestos; `outputs/` contiene modelos, métricas,
  predicciones, paquetes, previews y diagnósticos.
- Resolver raíces mediante `get_project_data_root()` y `get_output_root()`.

## Código

- `src/data/`: consolidación, revisión, finalización y splits YOLO-seg.
- `src/segmentation/`: adaptador de inferencia YOLO y quality gate.
- `src/preprocessing/`: geometría, máscaras, ROI y letterbox del segmentador.
- `src/training/segmentation_preflight.py`: gates previos a entrenamiento.
- `cloud_training/` y `modal_training.py`: plano de control remoto.

La configuración del runtime de inferencia está centralizada en
`config/segmentation.yaml`. El `dataset.yaml` dentro del dataset YOLO es un artefacto
portable generado, no la configuración global del proyecto.

## Seguridad

Smoke, entrenamiento y reanudación exigen confirmaciones explícitas separadas. No
relajar esos guards ni borrar `outputs/` sin confirmación literal.

## Verificación

```bash
python -m pytest
make lint
make check
```
