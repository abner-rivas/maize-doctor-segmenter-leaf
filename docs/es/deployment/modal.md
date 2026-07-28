# Entrenamiento en Modal

[Modal](https://modal.com/docs/guide) es la plataforma de GPU en la nube que usa el
proyecto para entrenar los **baselines**, correr **explicabilidad** y ejecutar el flujo
congelado de **segmentación foliar**. La ejecución se factura mientras dura y la
instancia se destruye al terminar.

## Cómo está montado

El flujo de baselines y explicabilidad comparte una imagen definida en
`scripts/modal/_common.py`. Esa factorización es deliberada: `train.py` y `explain.py`
necesitan exactamente la misma versión de torch y los mismos extras instalados para
que los checkpoints generados por uno se puedan leer desde el otro.

El almacenamiento persistente se resuelve con tres Volumes de Modal, que se crean solos la primera vez que se usan:

- `corn-clean`, montado en `/data`, contiene el dataset limpio.
- `corn-project-data`, montado en `/project-data`, contiene los splits derivados.
- `corn-outputs`, montado en `/outputs`, contiene pesos, métricas y reportes.

El dataset no se sube en cada corrida: se sube una única vez al volumen con `make modal-seed`, que es idempotente (si ya existe, no lo vuelve a descargar). Antes de esa primera vez hace falta autenticar la cuenta y darle a Modal acceso al dataset de Hugging Face:

```bash
pip install -e ".[cloud]"        # incluye el cliente modal
modal setup                      # autentica tu cuenta de Modal en el navegador
modal secret create hf HF_TOKEN=hf_xxxxxxxx   # token de Hugging Face para el dataset
```

## Cómo se corre el flujo

Con el dataset ya sembrado, entrenar un baseline en Modal se ve casi igual que entrenarlo en local, solo que con el prefijo `modal-`. Por ejemplo, para entrenar `efficientnet_b0` en la GPU A10, que es la que está configurada por defecto:

```bash
make modal-train-baselines MODELS=efficientnet_b0 EPOCHS=30
```

Las mismas banderas que acepta `train_baselines.py` en local están disponibles aquí, por ejemplo `NO_CAP=1` para entrenar sin tope de imágenes por clase, o `LIME=1` para que el run termine generando también los reportes LIME:

```bash
make modal-train-baselines MODELS=efficientnet_b0 NO_CAP=1
make modal-train-baselines MODELS=efficientnet_b0 LIME=1
```

Por debajo, `scripts/modal/train.py` traduce estas variables de `make` a los flags reales del script (`--models`, `--epochs`, `--no-cap`/`--max-per-class`, `--batch-size`, `--image-size`, `--learning-rate`, `--weight-decay`, `--num-workers`, `--no-pretrained`, `--lime`, `--regenerate-splits`), así que cualquier combinación que funcione en local funciona igual aquí.

Una vez que hay un run entrenado, la explicabilidad post-hoc se corre por separado, apuntando al modelo que se quiere analizar:

```bash
make modal-explain-lime MODELS=efficientnet_b0
make modal-explain-report MODELS=efficientnet_b0 SAMPLE_SIZE=50
make modal-explain-errors MODELS=efficientnet_b0
```

Igual que con el entrenamiento, `scripts/modal/explain.py` espeja los flags de `explain_lime.py`/`explain_report.py` (`--run`, `--baseline`, `--sample-size`, `--num-samples`, `--errors-only`). La única diferencia a tener en cuenta es que `--image`/`--output` de `explain-lime` deben ser rutas dentro del contenedor (relativas a `/data` u `/outputs`), no del filesystem local.

Si en algún momento hace falta limpiar runs o reportes, `make modal-clean-outputs` vacía
`corn-outputs` sin tocar el dataset ni los splits.

Quien prefiera no pasar por `make` puede invocar los mismos comandos de Modal directamente:

```bash
modal run scripts/modal/train.py::seed_dataset
modal run scripts/modal/train.py --models "efficientnet_b0" --epochs 30
modal run scripts/modal/explain.py::explain_report --models "efficientnet_b0" --sample-size 50
modal run scripts/modal/train.py::clean_outputs
```

## Segmentación foliar con el paquete congelado v2

La segmentación usa un control plane separado en `modal_training.py`:

- App `doctor-maiz-leaf-segmentation`.
- Volume persistente `doctor-maiz-leaf-segmentation`, montado en `/workspace`.
- Paquete inmutable
  `doctor_maiz_leaf_segmentation_cloud_v2-c087af60-seed42.tar.gz`, con SHA-256
  `4886ef3a11edb5d4819b9e980981a3f697f85129238a0b25e78eb9b0bc82805c`.
- Proyecto extraído en `/workspace/project` y paquete de entrada en
  `/workspace/incoming`.

El archivo de 2.13 GB no forma parte de la Image y no se monta con
`add_local_dir`. Se sube una sola vez al Volume. El adaptador local tampoco altera ni
reconstruye el paquete: el SHA anterior continúa siendo la identidad del release.

La Image de segmentación se construye una vez con Python 3.11, PyTorch 2.6.0,
torchvision 0.21.0 y `ultralytics==8.4.104`. La base está fijada por tag y por el
[digest publicado en Docker Hub](https://hub.docker.com/layers/pytorch/pytorch/2.6.0-cuda12.4-cudnn9-runtime/images/sha256-77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee):

```text
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@
sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
```

Las dependencias, `make`, Git, Bash, `sha256sum`, `tar` y las utilidades de
monitorización se instalan al construir la Image. Las funciones remotas no crean
`.venv-cloud`, no ejecutan el bootstrap y no reinstalan dependencias.

### Crear y cargar el Volume

Estos comandos presuponen que el cliente Modal ya está instalado y autenticado. El
adaptador se verificó contra Modal 1.5.3; este cambio no instala ni actualiza el
cliente.

```bash
modal volume create doctor-maiz-leaf-segmentation
modal volume put doctor-maiz-leaf-segmentation \
  outputs/leaf_detection/packages/doctor_maiz_leaf_segmentation_cloud_v2-c087af60-seed42.tar.gz \
  /incoming/
modal volume put doctor-maiz-leaf-segmentation \
  outputs/leaf_detection/packages/doctor_maiz_leaf_segmentation_cloud_v2-c087af60-seed42.tar.gz.sha256 \
  /incoming/
```

No se usa `--force` al subir: una colisión debe revisarse, no sobrescribirse
silenciosamente. `prepare` verifica el SHA-256, inspecciona el tar y extrae
atómicamente en `/workspace/project`. Una extracción existente sólo se reutiliza si
su marcador `.modal_package_prepared.json`, versión, fingerprint y checksums
coinciden.

### Orden de ejecución

Los comandos directos son:

```bash
modal run modal_training.py::prepare
modal run modal_training.py::preflight
modal run modal_training.py::smoke --confirm true
modal run --detach modal_training.py::train --confirm true
modal run --detach modal_training.py::resume --confirm true
modal run modal_training.py::validate
modal run modal_training.py::results
modal run modal_training.py::checksums
```

`train` y `resume` usan
[`--detach`](https://modal.com/docs/reference/cli/run) para que la ejecución no dependa
de la terminal local. No son intercambiables: `train` inicia un run nuevo con la
configuración final de 150 épocas y bloquea cualquier directorio existente; `resume`
sólo acepta el `last.pt` exacto identificado por `active_run_manifest.json` y deja un
manifiesto histórico de reanudación.

La GPU predeterminada es A10. Sólo se aceptan A10, L4 y A100, sin fallback automático
a una GPU más cara:

```bash
DOCTOR_MAIZ_MODAL_GPU=L4 modal run modal_training.py::preflight
DOCTOR_MAIZ_MODAL_GPU=A100 modal run --detach modal_training.py::train --confirm true
```

El preflight registra GPU solicitada y recibida, VRAM libre/total, uso inicial,
driver, CUDA, cuDNN, versiones de Python/PyTorch/torchvision/Ultralytics, digest base,
ID de Image cuando Modal lo expone, hash de receta y locks de dependencias. Bloquea
una GPU con menos de 12 GiB y sólo termina con
`ready_for_smoke_training`.

El smoke exige el literal `--confirm true`. Persiste `smoke_summary.json`,
`modal_smoke_manifest.json`, `best.pt`, `last.pt`, batch seleccionado, pico de VRAM,
duración y `train_yolo26n_seg.final.yaml`. El entrenamiento completo y la reanudación
también exigen `--confirm true`.

Cada función hace `reload()` al entrar y `commit()` al salir. Además, los Volumes
[sincronizan cambios en segundo plano](https://modal.com/docs/guide/volumes), por lo
que los checkpoints escritos durante un entrenamiento largo quedan en el
almacenamiento persistente antes del commit final.

No existe función Modal de test interno ni de piloto. `validate` evalúa
exclusivamente `best.pt` sobre `val`.

### Los mismos pasos con Make

El Makefile invoca `$(PYTHON) -m modal`. En este repositorio el cliente 1.5.3 está en
`.venv`, por lo que se puede seleccionar explícitamente con
`PYTHON=.venv/bin/python`:

```bash
make leaf-segmentation-modal-volume-create PYTHON=.venv/bin/python
make leaf-segmentation-modal-upload PYTHON=.venv/bin/python
make leaf-segmentation-modal-prepare PYTHON=.venv/bin/python
make leaf-segmentation-modal-preflight PYTHON=.venv/bin/python

CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 \
  make leaf-segmentation-modal-smoke PYTHON=.venv/bin/python
CONFIRM_SEGMENTATION_TRAINING=1 \
  make leaf-segmentation-modal-train PYTHON=.venv/bin/python
CONFIRM_SEGMENTATION_TRAINING=1 \
  make leaf-segmentation-modal-resume PYTHON=.venv/bin/python

make leaf-segmentation-modal-validate PYTHON=.venv/bin/python
make leaf-segmentation-modal-results PYTHON=.venv/bin/python
make leaf-segmentation-modal-checksums PYTHON=.venv/bin/python
make leaf-segmentation-modal-download PYTHON=.venv/bin/python
```

Para L4 o A100 se agrega `MODAL_SEGMENTATION_GPU=L4` o
`MODAL_SEGMENTATION_GPU=A100`. Los guards locales aceptan exactamente
`CONFIRM_SEGMENTATION_SMOKE_TRAINING=1` para smoke y
`CONFIRM_SEGMENTATION_TRAINING=1` para train/resume; después el adaptador remoto vuelve
a validar el literal `true`.

## Cómo se traen los resultados

Los resultados de cada corrida se versionan igual que en local, en `/outputs/baselines/<modelo>/<run_id>/` (donde `run_id` es un timestamp), así que un mismo modelo puede acumular varios runs sin pisarse entre sí. Para bajarlos a la máquina local:

```bash
make modal-pull            # copia el volumen corn-outputs -> ./outputs-remote
```

Por debajo esto es `modal volume get --force corn-outputs / ./outputs-remote`; el `--force` sobreescribe la carpeta local si ya existía de una corrida anterior.

## Otras notas útiles

Los splits del baseline se generan la primera vez que se necesitan en `corn-project-data` y
se reutilizan (lazy) en corridas siguientes. Si ya existen con otro tope, no se regeneran
solos: use `REGEN_SPLITS=1` para reemplazar exclusivamente ese split.

Para cambiar la GPU de los baselines todavía hay que editar `gpu="A10"` en
`scripts/modal/train.py`/`explain.py`. La segmentación no requiere editar código: use
`MODAL_SEGMENTATION_GPU` con la allowlist documentada arriba.
