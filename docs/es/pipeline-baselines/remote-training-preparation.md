# Preparación para entrenamiento remoto

Esta guía prepara y valida DoctorMaiz antes de usar una GPU remota. Ningún comando de las secciones de validación inicia épocas, optimizadores o descargas de pesos. No se debe entrenar mientras la auditoría, los splits o el preflight tengan bloqueos.

## Estructura esperada

En el servidor, separa código, dataset y resultados persistentes:

```text
/srv/doctor-maiz/                  # repositorio
├── config/
├── src/
├── scripts/
├── data/splits/seed_42_baseline/
├── pyproject.toml
├── Makefile
└── .env
/mnt/datasets/maize_dataset/data/  # DATASET_ROOT
└── clean/<clase>/{lab,real}/
/mnt/results/doctor-maiz/          # OUTPUT_ROOT opcional
/mnt/results/doctor-maiz/data/     # PROJECT_DATA_ROOT para splits derivados
```

El dataset no forma parte del manifiesto de código. Debe transferirse o montarse por separado, conservando exactamente `clean/<clase>/{lab,real}`.

## Entorno e instalación

```bash
cd /srv/doctor-maiz
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Comprueba que la versión de PyTorch instalada sea compatible con el runtime CUDA del proveedor. El proyecto declara los rangos de dependencias en `pyproject.toml`; no descargues pesos de modelos durante el preflight.

Crea `.env` a partir de la plantilla:

```bash
cp .env.example .env
```

Configura las raíces reales del servidor, sin agregar sufijos automáticamente:

```dotenv
DATASET_ROOT=/mnt/datasets/maize_dataset/data
OUTPUT_ROOT=/mnt/results/doctor-maiz
PROJECT_DATA_ROOT=/mnt/results/doctor-maiz/data
```

Si omites `OUTPUT_ROOT`, se usa `<repositorio>/outputs`. Si omites
`PROJECT_DATA_ROOT`, los splits viven en `<repositorio>/data/splits/seed_42_baseline/`.

## Archivos que deben copiarse

El manifiesto reproducible se genera con:

```bash
python scripts/checks/build_training_package_manifest.py \
  --output outputs/training_package_manifest.json
```

Incluye `config/`, `src/`, `scripts/`, los splits baseline, `pyproject.toml`, `Makefile` y `.env.example`. Excluye dataset, `.git`, entornos virtuales, cachés, ejecuciones abortadas y checkpoints. El JSON contiene tamaño y SHA-256 de cada archivo para comprobar la transferencia.

## Validaciones previas

Valida primero los splits oficiales:

```bash
python scripts/checks/validate_splits.py \
  --splits-dir data/splits/seed_42_baseline \
  --config config/dataset.yaml \
  --output data/splits/seed_42_baseline \
  --fail-on-error
```

Ejecuta después el preflight remoto:

```bash
python scripts/checks/training_preflight.py \
  --splits-dir data/splits/seed_42_baseline \
  --config config/dataset.yaml \
  --models efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0 \
  --device cuda \
  --check-dataset \
  --check-gpu \
  --output outputs/preflight
```

`preflight_report.json` es la evidencia procesable y `preflight_report.txt` el resumen humano. `LISTO` significa que no hay bloqueos. `BLOQUEADO` termina con código distinto de cero; corrige rutas, configuración, permisos, modelos, splits o disponibilidad CUDA antes de entrenar. Las advertencias deben revisarse, pero una consulta GPU sin GPU sólo es bloqueante cuando `--device cuda` fue solicitado.

También puede verificarse el cargador sin pesos externos ni entrenamiento:

```bash
python scripts/checks/smoke_loader.py \
  --splits-dir data/splits/seed_42_baseline \
  --batch-size 2 \
  --batches 2 \
  --check-sampler \
  --models efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0 \
  --device cpu
```

El smoke test sólo construye los modelos con `pretrained=False`; no ejecuta `forward`, `backward`, optimizador, épocas ni checkpoints.

## Comando de entrenamiento — ejecutar únicamente en el servidor

Después de un preflight `LISTO`, los argumentos confirmados del entrenador son:

```bash
python scripts/pipeline/train_baselines.py \
  --models efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0 \
  --baseline \
  --epochs 30
```

Para un solo modelo:

```bash
python scripts/pipeline/train_baselines.py \
  --models efficientnet_b0 \
  --baseline \
  --epochs 30
```

Estos bloques son instrucciones para la fase posterior; no forman parte de la preparación local. La ejecución directa de Python permanece disponible para automatización remota. Por Make, el seguro explícito exige:

```bash
make train-baselines CONFIRM_TRAINING=1
```

Sin `CONFIRM_TRAINING=1`, Make se detiene antes de invocar el script. `splits`, `splits-baseline`, `smoke-loader` y las validaciones no están bloqueados.

## Resultados, reinicio y recuperación

Las corridas se guardan en `OUTPUT_ROOT/baselines/<modelo>/<run_id>/`. Conserva `best.pth`, `last.pth`, `summary.json`, `train_history.csv`, `predictions.csv` y reportes asociados en almacenamiento persistente. Para descargarlos:

```bash
rsync -av usuario@servidor:/mnt/results/doctor-maiz/baselines/ ./outputs-remote/baselines/
```

El entrenador actual no expone un argumento de reanudación de checkpoint. No presentes una corrida parcial como reanudada: conserva la evidencia en `outputs/aborted_runs/`, revisa `ABORTED_RUN.md` y comienza un `run_id` nuevo. Una corrida abortada carece de cierre válido (`summary.json` y evaluación completa) y no debe entrar en comparaciones oficiales.

Para evitar entrenamiento local, usa localmente sólo auditoría, validación, smoke test CPU y preflight CPU. Reserva los dos comandos de entrenamiento anteriores para el servidor aprobado.
