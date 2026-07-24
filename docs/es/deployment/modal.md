# Entrenamiento en Modal

[Modal](https://modal.com/docs/guide) es la plataforma de GPU en la nube que usa el proyecto para entrenar los **baselines** y correr **explicabilidad** cuando conviene más potencia o más tiempo del que da la máquina local. La idea es simple: defines código Python normal, lo decoras para que Modal sepa qué correr en la nube, y esa ejecución se factura por segundo mientras dura. Cuando termina, la instancia se destruye sola (auto-teardown), así que no hay que acordarse de apagar nada ni pagar por GPU ociosa. Otra ventaja práctica es que los scripts de Modal exponen la misma CLI que sus equivalentes locales (`train_baselines.py`, `explain_lime.py`, `explain_report.py`): cualquier combinación de flags que funcione en local funciona igual en Modal.

## Cómo está montado

Todo el código que corre en Modal comparte una única imagen definida en `scripts/modal/_common.py`. Esa factorización es deliberada: `train.py` y `explain.py` necesitan exactamente la misma versión de torch y los mismos extras instalados para que los checkpoints generados por uno se puedan leer desde el otro. La imagen instala `torch==2.12.1` y `torchvision==0.27.1` desde el índice de PyTorch, más los extras `cloud` y `xai` del proyecto, y monta `src/` y `scripts/` en caliente para no tener que reconstruir la imagen en cada cambio de código.

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

Para cambiar de GPU basta con editar `gpu="A10"` en `scripts/modal/train.py`/`explain.py`; las opciones disponibles incluyen T4, L4, A10, L40S, A100 y H100.

Por ahora Modal solo cubre los baselines y su explicabilidad: el pipeline principal (`train.py`) todavía no está integrado.
