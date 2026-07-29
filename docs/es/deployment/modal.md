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

## Estado actual de la segmentación foliar

La segmentación usa el control plane separado `modal_training.py`, la App
`doctor-maiz-leaf-segmentation` y el Volume persistente del mismo nombre,
montado en `/workspace`.

Hay dos identidades que no deben confundirse:

- `/workspace/project_v4-7a4a5c08-seed42` conserva el entrenamiento aprobado y
  sus checkpoints;
- `/workspace/project_v5-test-7a4a5c08-seed42` contiene el código y dataset
  preparados para corregir la evaluación final.

El paquete v5 de validación es
`doctor_maiz_leaf_segmentation_cloud_v5-test-7a4a5c08-seed42.tar.gz`, con
SHA-256
`1ff54bbf56d0a5724bc472d56c5ea71192b9005b88b2dec89494ccb3dce59a79`.
Dos construcciones independientes produjeron exactamente ese mismo hash y
tamaño de 2 133 026 210 bytes.

La Image remota fija Python 3.11, PyTorch 2.6.0+cu124, torchvision
0.21.0+cu124, Ultralytics 8.4.104 y `faster-coco-eval==1.7.2`. Las
dependencias se instalan al construir la Image; ninguna función de evaluación
ejecuta `pip install`.

### Baseline congelado

El entrenamiento completo terminó el 2026-07-29:

- `status=passed`;
- 150 épocas, batch efectivo 26;
- duración 1 461,08 segundos;
- `best.pt` de 6 546 902 bytes, SHA-256
  `4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f`;
- `last.pt` de 6 546 902 bytes, SHA-256
  `b355074c7fb9afccf05db7e97535188f37ad5b5b5bbcc27ccf1a7d3e2f79a197`.

`best.pt`, `last.pt`, `training_summary.json`, `active_run_manifest.json`,
la configuración final y `results.csv` permanecen bajo
`project_v4-7a4a5c08-seed42/outputs/leaf_detection/segmenter/`.

### Estado de la evaluación final

`validate` solicita `split=test` de forma explícita y comprueba el split
efectivo después de que Ultralytics construye el dataloader. El intento
auditado sí escaneó `labels/test`, usó 173 imágenes y no usó `val` ni el
piloto.

La evaluación no está aprobada. Ultralytics redujo las 183 anotaciones
canónicas a 182 instancias porque los dos polígonos distintos de
`cldc_ec40ec2d7da5243e.txt` comparten el mismo bbox `float32` y su checker los
clasifica como duplicados. No existe `test_summary.json` con
`status=passed`, y las métricas del intento no son oficiales.

El 2026-07-29 se eliminaron del Volume todas las salidas no oficiales:

- `yolo26n_seg_val` y `yolo26n_seg_val-2`;
- sus dos directorios de predicciones;
- `val_summary.json`;
- el directorio `yolo26n_seg_test` incompleto;
- el checksum derivado del intento y `labels/test.cache`.

Por tanto, `segmenter_evaluation/` no existe actualmente. No debe volver a
ejecutarse `validate` hasta resolver formalmente la diferencia entre 183
anotaciones canónicas y 182 instancias efectivas sin alterar el fingerprint
test
`046545351ce79431bb1a995dfbc7dfa44c642a18a046860ed5edb9fc0ed89c51`.

### Comandos administrativos

El cliente Modal se usa desde `.venv`; `.venv-modal/` quedó retirado por ser
un entorno duplicado. Los comandos que no entrenan son:

```bash
make leaf-segmentation-modal-prepare PYTHON=.venv/bin/python
make leaf-segmentation-modal-preflight PYTHON=.venv/bin/python
make leaf-segmentation-modal-results PYTHON=.venv/bin/python
make leaf-segmentation-modal-checksums PYTHON=.venv/bin/python
make leaf-segmentation-modal-download PYTHON=.venv/bin/python
```

No se debe ejecutar `train`, `resume`, piloto ni una nueva evaluación final
mientras el gate de 183/182 permanezca abierto.

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
