# Entrenamiento cloud del segmentador

El entrenamiento de `yolo26n-seg.pt` queda reservado a una máquina CUDA
remota. El equipo local sólo construye y valida un paquete reproducible; no
instala Ultralytics, no descarga pesos y no ejecuta forward ni épocas.

## Contenido

`cloud_training/` contiene:

- bootstrap con resolución `pip --dry-run`, constraints dinámicos y bloqueo si
  se intenta sustituir torch o torchvision;
- preflight GPU/modelo que vuelve a verificar locks, fingerprints canónicos y
  dataset, y ejecuta un forward sintético del head de segmentación sin depender
  de que una imagen produzca detecciones;
- smoke de una época protegido por
  `CONFIRM_SEGMENTATION_SMOKE_TRAINING=1`;
- entrenamiento y reanudación protegidos por
  `CONFIRM_SEGMENTATION_TRAINING=1`;
- validación sobre val y evaluación separada sobre test interno;
- configs YAML, manifiesto del paquete y checksums.

El payload incluye únicamente `images/{train,val,test}`,
`labels/{train,val,test}`, `dataset.yaml`, los cinco manifiestos necesarios,
código, scripts y documentación mínima. Excluye `all/`, fuentes externas,
piloto, ZIP, outputs históricos, checkpoints, notebooks, entornos y cachés.

El piloto tiene un manifiesto de transporte separado y no forma parte del
archivo de entrenamiento.

## Flujo remoto

```bash
make leaf-segmentation-cloud-bootstrap
make leaf-segmentation-cloud-preflight

CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 \
  make leaf-segmentation-cloud-smoke

# Revise el batch medido y congele cualquier ajuste de workers/cache en este YAML.
CONFIRM_SEGMENTATION_TRAINING=1 \
  make leaf-segmentation-cloud-train \
  CONFIG=outputs/leaf_detection/segmenter/configs/train_yolo26n_seg.final.yaml

make leaf-segmentation-cloud-validate
make leaf-segmentation-cloud-test
```

El test interno tiene un gate de un solo uso: `evaluate_test.sh` aborta si
`test_summary.json` ya existe. Sólo una decisión formal registrada justifica
`FORCE_INTERNAL_TEST_RERUN=1`. La validación sobre `val` no tiene ese gate
porque es el nivel de desarrollo y debe poder repetirse.

Cada reanudación escribe su propio `resume_manifest_<timestamp>.json` además de
la copia con nombre estable, de modo que una corrida interrumpida varias veces
conserva el historial completo.

El preflight puede resolver los pesos en la nube, registra su ruta, tamaño,
SHA-256, fuente y versión exacta de Ultralytics, verifica el task y el head de
segmentación con un tensor CUDA sintético y se detiene antes del entrenamiento.
El smoke usa `batch=-1` para medir AutoBatch y sólo declara éxito si el batch
efectivo es un entero positivo, las pérdidas/métricas son finitas, la GPU fue
usada y `last.pt` existe con hash. La configuración final se crea una vez bajo
`outputs/leaf_detection/segmenter/configs/`; nunca se sobrescribe.

Antes de iniciar las 150 épocas, el runner crea
`active_run_manifest.json` con la identidad, configuración, fingerprints y
rutas esperadas del run. Así una interrupción se reanuda desde el `last.pt`
exacto y nunca desde un directorio incrementado implícitamente.

La reanudación nunca es automática:

```bash
CONFIRM_SEGMENTATION_TRAINING=1 \
  make leaf-segmentation-cloud-resume
```

## Construcción local permitida

```bash
make leaf-segmentation-cloud-package
```

El empaquetador usa lista blanca, orden estable, UID/GID cero, permisos
normalizados y mtime cero. Genera el `.tar.gz`, su `.sha256`, extrae en un
temporal, valida cada checksum y confirma que ninguna ruta excluida aparezca.

No se modifica `baseline_full` ni `leaf_detection.enabled=false`. No se usa el
piloto hasta una fase externa posterior.
