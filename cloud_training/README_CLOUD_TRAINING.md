# Entrenamiento cloud del segmentador

Este paquete contiene únicamente los splits congelados, código mínimo y gates
para entrenar `yolo26n-seg.pt` en una máquina CUDA. El piloto externo no forma
parte de train, val ni test interno.

## Procedimiento

1. Extraer el `.tar.gz` en almacenamiento persistente.
2. Confirmar que el Python provisto por la plataforma importa torch y
   torchvision con CUDA; no activar un virtualenv ajeno al paquete.
3. Ejecutar `bash cloud_training/bootstrap_cloud.sh`.
4. Ejecutar `bash cloud_training/preflight_cloud.sh`.
5. Revisar `outputs/leaf_detection/cloud_preflight/summary.json`.
6. Autorizar y ejecutar:
   `CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 bash cloud_training/smoke_train.sh`.
7. Revisar `outputs/leaf_detection/segmenter/smoke_summary.json` y el batch.
8. Autorizar el entrenamiento:
   `CONFIRM_SEGMENTATION_TRAINING=1 CONFIG=outputs/leaf_detection/segmenter/configs/train_yolo26n_seg.final.yaml bash cloud_training/train.sh`.
9. Ejecutar `bash cloud_training/validate.sh`.
10. Ejecutar `bash cloud_training/evaluate_test.sh`.
11. Descargar `outputs/leaf_detection/` antes de apagar la máquina.

Para reanudar de forma manual:

```bash
CONFIRM_SEGMENTATION_TRAINING=1 \
  bash cloud_training/resume_train.sh --reason interruption
```

## Plataforma genérica

Use una imagen con Python 3.11/3.12, PyTorch y torchvision CUDA ya compatibles,
una GPU con al menos 12 GiB de VRAM y un volumen persistente. El bootstrap no
reinstala PyTorch: fija sus versiones como constraints, simula la resolución de
Ultralytics y se bloquea si pip pretende sustituir torch o torchvision.

`batch=-1` es sólo el punto inicial remoto. El smoke registra el batch resuelto
y crea una configuración final bajo `outputs/leaf_detection/segmenter/configs/`.

No ejecute el piloto externo en esta fase. No hay entrenamiento automático:
smoke, entrenamiento y reanudación exigen variables distintas y explícitas.

## Comandos rápidos con Make

### En local

```bash
make leaf-segmentation-status
make leaf-segmentation-cloud-prepare
```

### En la nube

```bash
make leaf-segmentation-cloud-bootstrap
make leaf-segmentation-cloud-preflight

CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 \
make leaf-segmentation-cloud-smoke

CONFIRM_SEGMENTATION_TRAINING=1 \
make leaf-segmentation-cloud-train \
CONFIG=outputs/leaf_detection/segmenter/configs/train_yolo26n_seg.final.yaml

make leaf-segmentation-cloud-validate
make leaf-segmentation-cloud-test
make leaf-segmentation-cloud-results
```

### Reanudar

```bash
CONFIRM_SEGMENTATION_TRAINING=1 \
make leaf-segmentation-cloud-resume
```
