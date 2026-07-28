# Preflight de entrenamiento del segmentador

## Resultado

El preflight se ejecutó únicamente como auditoría y smoke test. Su estado es:

`blocked_by_missing_dependency`

Los bloqueos registrados son:

1. `missing_dependency`: Ultralytics no está instalado;
2. `missing_weights`: `yolo26n-seg.pt` no existe localmente;
3. `no_gpu`: CUDA no está disponible y `nvidia-smi` no se comunica con el
   driver.

La falta de GPU local no invalida el dataset ni impide preparar entrenamiento
remoto. La dependencia y los pesos sí requieren autorización explícita antes
de continuar.

## Locks y dataset

Se verificaron:

- `dataset_lock.status=ready_for_split_generation`;
- `split_lock.status=ready_for_training_preflight`;
- padre:
  `c087af60c2bad1c133c4ea8b14cee945405bfe4976aa80c4faf089d7a4b9e38c`;
- train:
  `6aa0bd03098137999f0ee9753a9128939ac123004513b1f7c5655d34c0fdd9df`;
- val:
  `3c7bf7aba8a9f29b409c61bad4d9e9d59a3387915592f181ad3950ac8374e720`;
- test:
  `046545351ce79431bb1a995dfbc7dfa44c642a18a046860ed5edb9fc0ed89c51`.

La revisión completa encontró 809/173/173 imágenes, 858/183/183 máscaras,
1 224 polígonos de clase única `0 = maize_leaf`, cero bbox mezclados, cero
archivos vacíos y cero errores de geometría o coordenadas. Las rutas de
`dataset.yaml` son relativas y el piloto no aparece en la configuración.

## Entorno local

| Componente | Resultado |
|---|---|
| SO | Linux 7.0.0-28-generic, x86_64, glibc 2.39 |
| Python | 3.12.3, `.venv` activa |
| pip | 24.0 |
| PyTorch | 2.12.1+cu130 |
| torchvision | 0.27.1 |
| CUDA compilada | 13.0 |
| CUDA disponible | no |
| cuDNN | 92000 |
| GPU/VRAM/driver | no disponibles; falla de comunicación con `nvidia-smi` |
| RAM | 33 227 743 232 bytes; 22 375 329 792 disponibles al auditar |
| CPU | 24 |
| Disco libre | 566 948 659 200 bytes |
| Ultralytics | no instalado |

No se instaló ninguna dependencia. El comando propuesto, pero no ejecutado, es:

```bash
python -m pip install "ultralytics==8.4.104"
```

Antes de autorizarlo debe revisarse el resolver: el entorno actual usa
PyTorch 2.12.1+cu130, torchvision 0.27.1 y CUDA compilada 13.0, y pip podría
sustituir esos paquetes.

## Modelo candidato

El candidato solicitado es `yolo26n-seg`, tarea `segment`:

- pesos esperados: `yolo26n-seg.pt`;
- configuración esperada: `yolo26n-seg.yaml`;
- pesos locales: no;
- configuración local: no;
- descarga necesaria para pesos preentrenados: sí;
- forward ejecutado: no;
- compatibilidad de entrenamiento/exportación: no verificable localmente;
- alternativas soportadas por la versión instalada: no enumerables porque no
  hay una versión instalada.

No se inventa soporte a partir del nombre. La licencia se registra como
Ultralytics AGPL-3.0 o Enterprise, pero no pudo validarse contra metadata local;
debe revisarse antes de autorizar instalación o entrenamiento.

## Smoke loader

Con semilla 42 se cargaron 4 muestras de train, 2 de val y 2 de test mediante
el loader oficial. Los polígonos se rasterizaron, se aplicó letterbox a 640 y
se produjo un solo batch:

- imágenes: `[8, 3, 640, 640]`, `torch.float32`;
- máscaras: `[8, 1, 640, 640]`, `torch.float32`;
- NaN: 0;
- infinito: 0;
- previews: 8.

No se creó modelo ni se ejecutaron forward, backward, `optimizer.step` o
épocas.

## Configuraciones propuestas

### Local conservadora

Está limitada a smoke test, no a entrenamiento completo:

- `imgsz=640`;
- `batch=1`;
- `device=cpu`;
- `workers=2`;
- `cache=False`;
- `epochs=150`, `patience=30`, `optimizer=auto`, sólo como propuesta;
- `seed=42`, `deterministic=True`.

### Remota recomendada

- GPU CUDA con al menos 12 GiB de VRAM como punto inicial;
- `imgsz=640`;
- `batch=-1` para AutoBatch según VRAM real;
- `device=0`;
- `workers=8`;
- `cache=disk`;
- `epochs=150`;
- `patience=30`;
- `optimizer=auto`;
- `seed=42`;
- `deterministic=True`;
- proyecto `outputs/leaf_detection/segmentation_training`;
- nombre `yolo26n_seg_full_seed42`.

No se fija un batch local de entrenamiento porque no existe GPU medible.

## Comando generado, no ejecutado

```bash
test "${CONFIRM_SEGMENTATION_TRAINING:-0}" = "1" || \
  { echo "Entrenamiento bloqueado: use CONFIRM_SEGMENTATION_TRAINING=1"; exit 2; }
yolo segment train model=yolo26n-seg.pt \
  data=data/leaf_detection/detector_dataset/dataset.yaml task=segment \
  imgsz=640 batch=-1 epochs=150 patience=30 seed=42 deterministic=True \
  device=0 workers=8 cache=disk optimizer=auto \
  project=outputs/leaf_detection/segmentation_training \
  name=yolo26n_seg_full_seed42
```

El mismo guard protege al objetivo `make leaf-segmentation-cloud-train`.
El preflight se ejecuta con `make leaf-segmentation-preflight`.

Nota: esta configuración remota fue la propuesta del preflight local. La
configuración oficial del paquete cloud es
`cloud_training/configs/train_yolo26n_seg.yaml`, que difiere de forma
deliberada en `cache=false` (frente a `cache=disk` propuesto aquí),
`project=outputs/leaf_detection/segmenter` y `name=yolo26n_seg_baseline`. Si el
smoke muestra al DataLoader como cuello de botella, `cache` deberá decidirse
formalmente antes del entrenamiento completo.

## Evidencia

Los reportes, configuración, comando y previews están bajo
`outputs/leaf_detection/training_preflight/`.

Contadores finales de seguridad: cero modelos, forwards, backwards,
optimizers, épocas, checkpoints, instalaciones, pesos descargados y accesos a
internet. Train, val, test, `all/`, fuentes externas, piloto, decisiones y
checkpoints históricos no fueron modificados.
