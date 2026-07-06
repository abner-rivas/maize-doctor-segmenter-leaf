# Integración de Modal para baselines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar los baselines en GPU de Modal de forma reproducible, reutilizando el pipeline `src/`, sin SSH ni gestión manual de instancias, coexistiendo con vast.ai.

**Architecture:** Un módulo `scripts/modal/train.py` define una `modal.App`, una `Image` nativa (deps horneadas, código `src`/`scripts` montado en caliente) y dos entrypoints (`seed_dataset`, `train_baselines`) que **orquestan por subprocess los mismos scripts CLI que corre el Makefile**. El dataset vive en un Volume `corn-clean` (seed una vez); los artefactos en un Volume `corn-outputs`. `get_output_root()` se hace env-overridable (`OUTPUT_ROOT`) para redirigir salidas al Volume; `get_dataset_root()` ya es env-driven.

**Tech Stack:** Modal (Python SDK), PyTorch CUDA 12.6, el pipeline existente (`create_splits.py`, `train_baselines.py`, `download_dataset.py`), Make.

## Global Constraints

- Python **>=3.11** (imagen `debian_slim(python_version="3.11")`).
- Torch pineado: `torch==2.12.1`, `torchvision==0.27.1` desde `index_url=https://download.pytorch.org/whl/cu126`. Estos pines satisfacen los rangos de `pyproject.toml` (`torch>=2.2,<2.13`, `torchvision>=0.17,<0.28`) — mantener ese invariante.
- **`get_output_root()` debe seguir siendo backward-compatible**: sin `OUTPUT_ROOT`, devuelve `PROJECT_ROOT/outputs` (13 consumidores dependen de esto).
- **No introducir pytest.** Verificación por *smoke checks* (`python -c` con salida esperada), `ruff`, y una corrida PoC manual en Modal — convención del repo.
- **No tocar `scripts/vastai/`, `src/` (salvo `config.py`), ni `raw/`/`clean/`.**
- GPU por defecto: `A10`. Extra de dependencias: `cloud` (se le añade `modal`).
- Volumes: `corn-clean` → `/data`; `corn-outputs` → `/outputs`. Anclaje del repo en el container: `/root` (workdir por defecto de Modal).
- Secret de Modal: `hf` con `HF_TOKEN`. `HF_DATASET_REPO` se pasa como env no-secreto en la imagen.

---

### Task 1: `get_output_root` env-overridable + `.env.example`

**Files:**
- Modify: `src/config.py:12-14` (bloque de `_raw_dataset_root`), `src/config.py:27-30` (`get_output_root`)
- Modify: `.env.example` (añadir `OUTPUT_ROOT` documentado)

**Interfaces:**
- Consumes: nada de tareas previas.
- Produces: `get_output_root() -> Path` que devuelve `Path(os.environ["OUTPUT_ROOT"])` si la env está seteada y no vacía, si no `PROJECT_ROOT / "outputs"`. Firma sin cambios.

- [ ] **Step 1: Añadir lectura de `OUTPUT_ROOT` junto a `DATASET_ROOT`**

En `src/config.py`, después del bloque de `DATASET_ROOT` (línea 14), añadir:

```python
_raw_output_root = os.getenv("OUTPUT_ROOT", "").strip()
```

- [ ] **Step 2: Reescribir `get_output_root` para respetar la env**

Reemplazar el cuerpo actual de `get_output_root` (líneas 27-30) por:

```python
def get_output_root() -> Path:
    """OUTPUT_ROOT si está definido; si no, PROJECT_ROOT/outputs (default local).

    Env-overridable (mismo patrón que DATASET_ROOT) para redirigir artefactos a un
    volumen persistente en entornos remotos (p.ej. Modal). Sin OUTPUT_ROOT el
    comportamiento local no cambia."""
    return Path(_raw_output_root) if _raw_output_root else PROJECT_ROOT / "outputs"
```

- [ ] **Step 3: Documentar `OUTPUT_ROOT` en `.env.example`**

Añadir al final de `.env.example`:

```bash

# (Opcional) Raíz de artefactos generados (splits, resultados de entrenamiento, reports).
# Si se omite, el pipeline usa <repo>/outputs. Úsalo para redirigir a un volumen montado
# (p.ej. Modal: OUTPUT_ROOT=/outputs). No lo definas en local salvo que quieras moverlos.
# OUTPUT_ROOT=/outputs
```

- [ ] **Step 4: Verificar el default (sin `OUTPUT_ROOT`) — smoke check**

Run (desde la raíz del repo, con el venv activo):
```bash
venv/Scripts/python -c "import os; os.environ.pop('OUTPUT_ROOT', None); from src.config import get_output_root, PROJECT_ROOT; assert get_output_root() == PROJECT_ROOT / 'outputs', get_output_root(); print('OK default', get_output_root())"
```
Expected: imprime `OK default <repo>\outputs` sin AssertionError.

> Nota Windows/PowerShell: en cmd/PowerShell usa `venv\Scripts\python`; en bash `venv/Scripts/python`. En Linux/Mac: `venv/bin/python`.

- [ ] **Step 5: Verificar el override (`OUTPUT_ROOT` seteado) — smoke check**

Run (bash / git-bash):
```bash
OUTPUT_ROOT=/tmp/modal_out venv/Scripts/python -c "from pathlib import Path; from src.config import get_output_root; assert get_output_root() == Path('/tmp/modal_out'), get_output_root(); print('OK override', get_output_root())"
```
Run (PowerShell — el prefijo `VAR=val cmd` no existe en PowerShell):
```powershell
$env:OUTPUT_ROOT='/tmp/modal_out'; venv\Scripts\python -c "from pathlib import Path; from src.config import get_output_root; assert get_output_root() == Path('/tmp/modal_out'), get_output_root(); print('OK override', get_output_root())"; Remove-Item Env:\OUTPUT_ROOT
```
Expected: imprime `OK override /tmp/modal_out`.

> `_raw_output_root` se lee a nivel de módulo, así que la env debe estar seteada **antes** del import (como en el comando de arriba). Es intencional: espeja el patrón de `DATASET_ROOT`.

- [ ] **Step 6: Lint**

Run: `venv/Scripts/ruff check src/config.py`
Expected: `All checks passed!` (o sin errores nuevos).

- [ ] **Step 7: Commit**

```bash
git add src/config.py .env.example
git commit -m "feat(config): OUTPUT_ROOT env-overridable en get_output_root"
```

---

### Task 2: Añadir `modal` al extra `cloud`

**Files:**
- Modify: `pyproject.toml:20` (extra `cloud`)

**Interfaces:**
- Consumes: nada.
- Produces: `import modal` disponible tras `pip install -e ".[cloud]"`.

- [ ] **Step 1: Añadir `modal` al extra `cloud`**

En `pyproject.toml`, reemplazar la línea del extra `cloud` (línea 20):

```toml
cloud    = ["huggingface_hub>=0.24,<1.22", "gdown>=5.1,<6.2"]
```
por:
```toml
cloud    = ["huggingface_hub>=0.24,<1.22", "gdown>=5.1,<6.2", "modal>=0.64"]
```

- [ ] **Step 2: Instalar el extra**

Run: `venv/Scripts/pip install -e ".[cloud]"`
Expected: instala `modal` (y deps) sin errores; termina en `Successfully installed ... modal-...`.

- [ ] **Step 3: Verificar el import — smoke check**

Run: `venv/Scripts/python -c "import modal; print('modal', modal.__version__)"`
Expected: imprime `modal <versión>` (>=0.64).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: añade modal al extra cloud"
```

---

### Task 3: Módulo Modal `scripts/modal/train.py`

**Files:**
- Create: `scripts/modal/train.py`
- Create: `scripts/modal/__init__.py` (vacío — coherencia con `scripts/__init__.py`)

**Interfaces:**
- Consumes: `get_output_root` env-aware (Task 1); `modal` instalado (Task 2); los scripts CLI existentes `scripts/dataset/download_dataset.py`, `scripts/pipeline/create_splits.py`, `scripts/pipeline/train_baselines.py`.
- Produces:
  - `app`: `modal.App("corn-leaf-baselines")`.
  - `seed_dataset()`: función Modal CPU que descarga el dataset al Volume `corn-clean` y commitea.
  - `train_baselines(models: str = "efficientnet_b0", epochs: int = 30)`: función Modal GPU (`A10`) que genera splits lazy y entrena, commiteando a `corn-outputs`.
  - `main(models, epochs, seed)`: `local_entrypoint` para `modal run`.

- [ ] **Step 1: Crear `scripts/modal/__init__.py` vacío**

```python
```
(archivo vacío)

- [ ] **Step 2: Escribir el módulo `scripts/modal/train.py`**

```python
"""Entrenamiento de baselines en GPU de Modal (https://modal.com/docs/guide).

Coexiste con scripts/vastai/. No importa funciones internas del pipeline: orquesta por
subprocess los mismos scripts CLI que corre el Makefile (splits-baseline, train-baselines),
heredando el entorno de la imagen (DATASET_ROOT=/data, OUTPUT_ROOT=/outputs) para que
get_dataset_root()/get_output_root() resuelvan a los Volumes montados.

Uso:
    modal run scripts/modal/train.py::seed_dataset            # 1 vez: dataset -> Volume
    modal run scripts/modal/train.py --models "efficientnet_b0" --epochs 30
Requiere: `pip install -e ".[cloud]"`, `modal setup`, y el secret:
    modal secret create hf HF_TOKEN=hf_xxx
"""

import subprocess
import sys
from pathlib import Path

import modal

REPO_ANCHOR = "/root"  # workdir por defecto de Modal; el código local se monta aquí
HF_DATASET_REPO = "daiv05/corn-leaf-diseases-pests-and-deficiencies"

# Volumes persistentes: dataset (seed una vez) y artefactos (splits/pesos/métricas/LIME).
dataset_vol = modal.Volume.from_name("corn-clean", create_if_missing=True)
outputs_vol = modal.Volume.from_name("corn-outputs", create_if_missing=True)

# Imagen nativa: deps horneadas; src/scripts montados en caliente (última capa, copy=False).
# .env() va ANTES de los add_local_* porque las capas copy=False deben ser las últimas.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
        "torchvision==0.27.1",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["cloud", "xai"])
    .env(
        {
            "DATASET_ROOT": "/data",
            "OUTPUT_ROOT": "/outputs",
            "HF_DATASET_REPO": HF_DATASET_REPO,
        }
    )
    .add_local_dir("config", f"{REPO_ANCHOR}/config", copy=True)
    .add_local_python_source("src", "scripts")
)

app = modal.App("corn-leaf-baselines", image=image)


@app.function(
    volumes={"/data": dataset_vol},
    secrets=[modal.Secret.from_name("hf")],
    timeout=3600,
)
def seed_dataset() -> None:
    """Descarga el dataset limpio al Volume corn-clean. Idempotente: download_dataset.py
    salta si /data/clean ya tiene contenido."""
    subprocess.run(
        [sys.executable, "scripts/dataset/download_dataset.py"],
        check=True,
        cwd=REPO_ANCHOR,
    )
    dataset_vol.commit()


@app.function(
    gpu="A10",
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    timeout=6 * 3600,
)
def train_baselines(models: str = "efficientnet_b0", epochs: int = 30) -> None:
    """Genera splits baseline (lazy) y entrena los baselines indicados, persistiendo
    resultados en el Volume corn-outputs."""
    dataset_vol.reload()  # ve el dataset seedeado por seed_dataset
    splits_marker = Path("/outputs/splits/seed_42_baseline/train.csv")
    if not splits_marker.exists():
        subprocess.run(
            [sys.executable, "scripts/pipeline/create_splits.py", "--baseline"],
            check=True,
            cwd=REPO_ANCHOR,
        )
    subprocess.run(
        [
            sys.executable,
            "scripts/pipeline/train_baselines.py",
            "--models",
            *models.split(),
            "--baseline",
            "--epochs",
            str(epochs),
        ],
        check=True,
        cwd=REPO_ANCHOR,
    )
    outputs_vol.commit()


@app.local_entrypoint()
def main(models: str = "efficientnet_b0", epochs: int = 30) -> None:
    """Entrypoint de `modal run`: dispara train_baselines en la GPU remota."""
    train_baselines.remote(models=models, epochs=epochs)
```

- [ ] **Step 3: Verificar que el módulo define la app sin auth — smoke check**

Definir la `App`/`Image`/`Volume` NO requiere autenticación de Modal (solo `modal run`/`deploy` la piden). El import ejecuta los decoradores y construye la imagen en memoria.

Run: `venv/Scripts/python -c "import scripts.modal.train as m; print('app', m.app.name); print('fns', [f for f in ('seed_dataset','train_baselines') if hasattr(m, f)])"`
Expected: imprime `app corn-leaf-baselines` y `fns ['seed_dataset', 'train_baselines']` sin excepción.

> Si `modal.Volume.from_name(..., create_if_missing=True)` intentara contactar el backend y fallara sin auth, el import fallará: en ese caso, verificar solo la sintaxis con `venv/Scripts/python -m py_compile scripts/modal/train.py` (Expected: sin salida = OK) y dejar la validación funcional para el PoC (Task 6).

- [ ] **Step 4: Lint**

Run: `venv/Scripts/ruff check scripts/modal/train.py`
Expected: `All checks passed!` (o sin errores nuevos).

- [ ] **Step 5: Commit**

```bash
git add scripts/modal/train.py scripts/modal/__init__.py
git commit -m "feat(modal): módulo de entrenamiento de baselines en Modal"
```

---

### Task 4: Targets Make para Modal

**Files:**
- Modify: `Makefile:15` (línea `.PHONY`), y añadir bloque de targets tras `train-baselines-full` (línea ~36)

**Interfaces:**
- Consumes: `scripts/modal/train.py` (Task 3); variables `MODELS` (ya existe, default `all`) y una nueva `EPOCHS`.
- Produces: targets `modal-seed`, `modal-train-baselines`, `modal-pull`.

- [ ] **Step 1: Declarar la variable `EPOCHS` con default**

En `Makefile`, tras la línea `MODELS ?= all` (línea 13), añadir:

```makefile
EPOCHS ?= 30
```

- [ ] **Step 2: Registrar los targets nuevos en `.PHONY`**

En la línea `.PHONY:` (línea 15), añadir al final: `modal-seed modal-train-baselines modal-pull`.

- [ ] **Step 3: Añadir los targets tras `train-baselines-full`**

Insertar tras el bloque `train-baselines-full` (línea 36):

```makefile
modal-seed:
	modal run scripts/modal/train.py::seed_dataset

modal-train-baselines:
	modal run scripts/modal/train.py --models "$(MODELS)" --epochs "$(EPOCHS)"

modal-pull:
	modal volume get corn-outputs / ./outputs-remote
```

> `modal-train-baselines` con `MODELS=all` pasaría `--models all` (soportado por `train_baselines.py`). Para el PoC conviene un modelo concreto: `make modal-train-baselines MODELS=efficientnet_b0 EPOCHS=1`.

- [ ] **Step 4: Verificar que Make expande los comandos — smoke check**

Run: `make -n modal-seed modal-train-baselines modal-pull MODELS=efficientnet_b0 EPOCHS=1`
Expected (sin ejecutar Modal): imprime las tres líneas
```
modal run scripts/modal/train.py::seed_dataset
modal run scripts/modal/train.py --models "efficientnet_b0" --epochs "1"
modal volume get corn-outputs / ./outputs-remote
```

> En Windows sin `make`, verificar manualmente que el bloque quedó bien escrito (indentado con TAB, no espacios).

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "build(make): targets modal-seed/modal-train-baselines/modal-pull"
```

---

### Task 5: Documentación `docs/es/deployment/modal.md`

**Files:**
- Create: `docs/es/deployment/modal.md`
- Read (referencia de estilo, no modificar): `docs/es/deployment/vast-ai.md`

**Interfaces:**
- Consumes: los targets y el módulo de las tareas 3-4.
- Produces: guía de uso end-to-end.

- [ ] **Step 1: Leer la guía de vast.ai para espejar tono/estructura**

Run: abrir `docs/es/deployment/vast-ai.md` y notar secciones (requisitos, setup, flujo, notas de costo).

- [ ] **Step 2: Escribir `docs/es/deployment/modal.md`**

```markdown
# Entrenamiento en Modal

Guía para entrenar los **baselines** en GPU de [Modal](https://modal.com/docs/guide).
Modal coexiste con vast.ai (ver `vast-ai.md`); no lo reemplaza. A diferencia de vast.ai
(VM + SSH), en Modal defines código que corre en la nube y se cobra por segundo, con
auto-teardown (no hay que acordarse de destruir instancias).

## Requisitos (una sola vez)

```bash
pip install -e ".[cloud]"        # incluye el cliente modal
modal setup                      # autentica tu cuenta de Modal en el navegador
modal secret create hf HF_TOKEN=hf_xxxxxxxx   # token de Hugging Face para el dataset
```

## Volúmenes

Se crean solos la primera vez (`create_if_missing=True`):
- `corn-clean` → dataset limpio (montado en `/data`).
- `corn-outputs` → artefactos: splits, pesos, métricas, LIME (montado en `/outputs`).

## Flujo

```bash
# 1) Seed del dataset al volumen (una sola vez; idempotente)
make modal-seed

# 2) Entrenar baselines en GPU (A10 por defecto)
make modal-train-baselines MODELS=efficientnet_b0 EPOCHS=30
#   MODELS acepta uno o varios (separados por espacio) o "all".

# 3) Traer los resultados al equipo local
make modal-pull            # copia el volumen corn-outputs -> ./outputs-remote
```

Equivalente sin `make`:
```bash
modal run scripts/modal/train.py::seed_dataset
modal run scripts/modal/train.py --models "efficientnet_b0" --epochs 30
modal volume get corn-outputs / ./outputs-remote
```

## Notas

- Los resultados se versionan por corrida en `/outputs/baselines/<modelo>/<run_id>/`
  (igual que en local; `run_id` = timestamp). `make modal-pull` los baja a `./outputs-remote`.
- Los splits baseline se generan la primera vez y se reutilizan (lazy) en corridas siguientes.
- Para cambiar la GPU, edita `gpu="A10"` en `scripts/modal/train.py` (opciones: T4, L4, A10,
  L40S, A100, H100…).
- Solo cubre baselines; el pipeline principal (`train.py`) no está integrado.
```

- [ ] **Step 3: Verificar enlaces/paths mencionados**

Run: `venv/Scripts/python -c "from pathlib import Path; [print(p, Path(p).exists()) for p in ['scripts/modal/train.py','docs/es/deployment/vast-ai.md','docs/es/deployment/modal.md']]"`
Expected: las tres rutas `True`.

- [ ] **Step 4: Commit**

```bash
git add docs/es/deployment/modal.md
git commit -m "docs: guía de entrenamiento en Modal (baselines)"
```

---

### Task 6: PoC de aceptación en Modal (manual, requiere credenciales)

**Files:** ninguno (validación end-to-end).

**Interfaces:**
- Consumes: todo lo anterior + cuenta de Modal autenticada (`modal setup`) + secret `hf`.
- Produces: evidencia de que el flujo completo funciona (run dir en el Volume, descarga local).

> **Gate:** esta tarea requiere una cuenta de Modal y créditos. Si no están disponibles, marcarla como *pendiente de validación manual* y no bloquear el merge de las tareas 1-5 (que sí se verifican offline).

- [ ] **Step 1: Regresión local — el default de `get_output_root` no cambió**

Run (sin `OUTPUT_ROOT` en el entorno): `make splits-baseline` (o el smoke check de Task 1 Step 4).
Expected: los splits se escriben bajo `<repo>/outputs/splits/seed_42_baseline/` como antes.

- [ ] **Step 2: Seed del dataset**

Run: `make modal-seed`
Expected: logs de descarga; termina sin error. Segunda ejecución: salta la descarga (idempotente).

- [ ] **Step 3: Verificar el Volume del dataset**

Run: `modal volume ls corn-clean /`
Expected: lista `clean/` (o el layout del dataset).

- [ ] **Step 4: Entrenamiento corto (1 época, 1 modelo)**

Run: `make modal-train-baselines MODELS=efficientnet_b0 EPOCHS=1`
Expected: genera splits (primera vez), entrena 1 época en A10, termina sin error.

- [ ] **Step 5: Verificar artefactos en el Volume de outputs**

Run: `modal volume ls corn-outputs /baselines/efficientnet_b0`
Expected: existe un directorio `<run_id>` (timestamp) con pesos/métricas.

- [ ] **Step 6: Descargar resultados a local**

Run: `make modal-pull`
Expected: `./outputs-remote/baselines/efficientnet_b0/<run_id>/` presente localmente.

- [ ] **Step 7: Idempotencia de splits**

Run de nuevo: `make modal-train-baselines MODELS=efficientnet_b0 EPOCHS=1`
Expected: NO regenera splits (el marker `/outputs/splits/seed_42_baseline/train.csv` ya existe); va directo a entrenar.

---

## Notas de cierre

- Al terminar todas las tareas: `superpowers:finishing-a-development-branch` para decidir merge/PR de `feat/modal-baselines`.
- Riesgo abierto (documentado en el spec): las firmas exactas de `pip_install_from_pyproject` / `add_local_python_source` y el path de anclaje `/root` se confirman recién en el PoC (Task 6). Si Modal monta el código en otra ruta, ajustar `REPO_ANCHOR` y el `add_local_dir` de `config`.
