# Entrenamiento cloud del segmentador

El entrenamiento de `yolo26n-seg.pt` queda reservado a una máquina CUDA
remota. El equipo local sólo construye y valida un paquete reproducible; no
instala Ultralytics, no descarga pesos y no ejecuta forward ni épocas.

## Contenido

`cloud_training/` contiene:

- bootstrap con resolución `pip --dry-run`, constraints dinámicos y bloqueo si
  se intenta sustituir torch o torchvision;
- preflight GPU/modelo que vuelve a verificar locks, fingerprints y dataset;
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
bash cloud_training/bootstrap_cloud.sh
bash cloud_training/preflight_cloud.sh

CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 \
  bash cloud_training/smoke_train.sh

CONFIRM_SEGMENTATION_TRAINING=1 \
  bash cloud_training/train.sh

bash cloud_training/validate.sh
bash cloud_training/evaluate_test.sh
```

El preflight puede resolver los pesos en la nube, registra su ruta, tamaño,
SHA-256, fuente y versión de Ultralytics, y se detiene antes del entrenamiento.
El smoke registra duración, VRAM máxima, métricas, pérdidas disponibles,
checkpoint y batch seleccionado. Si AutoBatch resuelve un entero positivo,
se conserva una configuración final bajo
`outputs/leaf_detection/segmenter/configs/`.

La reanudación nunca es automática:

```bash
CONFIRM_SEGMENTATION_TRAINING=1 \
  bash cloud_training/resume_train.sh --reason interruption
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
