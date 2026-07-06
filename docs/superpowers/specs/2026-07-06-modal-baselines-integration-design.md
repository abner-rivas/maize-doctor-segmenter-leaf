# Diseño: integración de Modal para el pipeline de baselines

**Fecha:** 2026-07-06
**Alcance:** solo baselines. Modal coexiste con vast.ai (no lo reemplaza).

## Objetivo

Permitir entrenar los baselines (`scripts/pipeline/train_baselines.py`) en GPU de
[Modal](https://modal.com/docs/guide) de forma reproducible, sin SSH ni gestión manual de
instancias, reutilizando el pipeline de datos/modelos existente en `src/`.

## Contexto y restricciones

- El pipeline es Makefile-céntrico (`$(PYTHON) = venv/bin/python`) con un helper por CLI
  (`scripts/vastai/launch.py`) para GPU remota. Modal introduce un modelo distinto: código
  Python decorado que corre en la nube, sin SSH.
- `get_dataset_root()` ya es env-driven (`DATASET_ROOT`) → funciona en Modal apuntando al mount.
- `get_output_root()` está **hardcodeado** a `PROJECT_ROOT/outputs` (`src/config.py:27-30`).
  En Modal ese path es efímero; hay que redirigirlo a un Volume persistente.
- `create_splits()` ya es invocable como función (`scripts/pipeline/create_splits.py:68`).
- `train_baselines.main()` usa `parser.parse_args()` sin argv → no invocable programáticamente.
- 13 archivos consumen `get_output_root()` → cualquier cambio debe ser backward-compatible.

## Decisiones de diseño (acordadas)

1. **Imagen:** nativa de Modal (chaining) con código en caliente vía
   `add_local_python_source` — cambiar `src/` NO rebuildea la imagen. (Descartado:
   `Image.from_dockerfile`, que rebuildea en cada cambio y arrastra el patrón `venv/`.)
2. **Splits:** generación *lazy* dentro del entrypoint de training (idempotente). (Descartado:
   función `prepare_splits` separada que exige recordar correrla.)
3. **Ergonomía:** targets `make` delgados que envuelven `modal run`. (Descartado: solo
   documentar `modal run` sin tocar el Makefile.)
4. **Dependencia:** extra propio `modal` en `pyproject.toml` (no mezclar con `cloud`, que es
   descarga de dataset).
5. **GPU por defecto:** `A10` (VRAM/precio adecuados para efficientnet/mobilenet), overridable.

## Componentes

### 1. `get_output_root` env-overridable — `src/config.py`

Espejo del patrón de `DATASET_ROOT`. Lee `OUTPUT_ROOT`; si está, lo usa; si no, cae al
default actual. Backward-compatible: sin `OUTPUT_ROOT`, los 13 consumidores no cambian.

```python
_raw_output_root = os.getenv("OUTPUT_ROOT", "").strip()

def get_output_root() -> Path:
    """OUTPUT_ROOT si está definido; si no, PROJECT_ROOT/outputs (default local)."""
    return Path(_raw_output_root) if _raw_output_root else PROJECT_ROOT / "outputs"
```

**Interfaz:** misma firma `() -> Path`. **Dependencias:** `os.getenv`, `PROJECT_ROOT`.
Añadir `OUTPUT_ROOT` (comentado, opcional) a `.env.example`.

### 2. `train_baselines.main` invocable — `scripts/pipeline/train_baselines.py`

Cambio de una línea para permitir inyectar flags desde la función Modal, sin alterar el CLI:

```python
def main(argv: list[str] | None = None) -> None:
    ...
    args = parser.parse_args(argv)   # argv=None => sys.argv (comportamiento CLI actual)
```

`if __name__ == "__main__": main()` queda igual. El Makefile y el CLI no cambian.

### 3. Módulo Modal — `scripts/modal/train.py`

Define `App`, `Image`, dos Volumes y dos entrypoints.

**Volumes:**
- `corn-clean` → mount `/data` — dataset limpio (seed una vez).
- `corn-outputs` → mount `/outputs` — splits + pesos + métricas + LIME.

**Imagen (nativa):**
```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1",
                 index_url="https://download.pytorch.org/whl/cu126")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["cloud", "xai"])
    .add_local_dir("config", remote_path="/root/config")
    .add_local_python_source("src", "scripts")
    .env({"DATASET_ROOT": "/data", "OUTPUT_ROOT": "/outputs"})
)
```
`HF_TOKEN` vía `modal.Secret.from_name("hf")`.

> Los pines `torch==2.12.1` / `torchvision==0.27.1` satisfacen los rangos de `pyproject.toml`
> (`torch>=2.2,<2.13`, `torchvision>=0.17,<0.28`), así que `pip_install_from_pyproject` los deja
> como están y no re-descarga wheels CPU desde PyPI. Mantener ese invariante si se suben los pines.

> Nota de implementación: verificar contra la doc vigente de Modal la firma exacta de
> `pip_install_from_pyproject` y `add_local_python_source` (con Context7 / WebFetch) antes de
> fijar el código; los nombres pueden variar por versión.

**`seed_dataset` (CPU, 1 vez):**
```python
@app.function(image=image, volumes={"/data": dataset_vol},
              secrets=[modal.Secret.from_name("hf")], timeout=3600)
def seed_dataset():
    # idempotente: si /data/clean ya tiene contenido, no re-descarga
    if dataset_ya_poblado("/data"):
        return
    subprocess.run([sys.executable, "scripts/dataset/download_dataset.py"], check=True)
    dataset_vol.commit()
```

**`train_baselines` (GPU):**
```python
@app.function(image=image, gpu="A10",
              volumes={"/data": dataset_vol, "/outputs": outputs_vol},
              secrets=[modal.Secret.from_name("hf")], timeout=6 * 3600)
def train_baselines(models: str = "efficientnet_b0", epochs: int = 30):
    from scripts.pipeline import create_splits, train_baselines as tb
    if not splits_baseline_existen():        # /outputs/splits/seed_42_baseline/*.csv
        create_splits.create_splits(baseline=True)
        outputs_vol.commit()
    tb.main(["--models", *models.split(), "--baseline", "--epochs", str(epochs)])
    outputs_vol.commit()
```

`@app.local_entrypoint()` traduce args de `modal run` a la llamada remota.

### 4. Targets Make — `Makefile`

```makefile
modal-seed:
	modal run scripts/modal/train.py::seed_dataset

modal-train-baselines:
	modal run scripts/modal/train.py --models "$(MODELS)" --epochs "$(EPOCHS)"

modal-pull:
	modal volume get corn-outputs / ./outputs-remote
```

### 5. Dependencia — `pyproject.toml`

```toml
modal = ["modal>=0.64"]
```
(Confirmar versión mínima vigente al implementar.)

### 6. Documentación — `docs/es/deployment/modal.md`

Espejo de `docs/es/deployment/vast-ai.md`: setup (`pip install -e ".[modal]"`,
`modal setup`, `modal secret create hf HF_TOKEN=…`) y flujo
`make modal-seed` → `make modal-train-baselines MODELS=…` → `make modal-pull`.

## Flujo de datos

```
[local] modal run seed_dataset ──> descarga HF ──> Volume corn-clean (/data)   [1 vez]
[local] modal run train_baselines ─> monta /data + /outputs
                                     ├─ splits lazy ─> /outputs/splits/seed_42_baseline
                                     ├─ tb.main(--baseline) ─> /outputs/baselines/<modelo>/<run_id>
                                     └─ outputs_vol.commit()
[local] make modal-pull ──> modal volume get corn-outputs ──> ./outputs-remote
```

## Manejo de errores

- `seed_dataset`: idempotente; si el Volume ya tiene el dataset, retorna sin re-descargar.
- Falta de `HF_TOKEN`/Secret: falla en el arranque de la función con mensaje de Modal.
- `create_splits`/`train_baselines` propagan sus `SystemExit`/excepciones → la corrida de
  Modal termina en fallo visible en los logs; nada queda "colgado" cobrando (auto-teardown).
- `commit()` explícito tras escribir para asegurar persistencia aun si algo falla después.

## Testing / validación

- **PoC manual:** `make modal-seed` una vez; luego `make modal-train-baselines
  MODELS=efficientnet_b0 EPOCHS=1` en T4/A10; verificar que aparece
  `/outputs/baselines/efficientnet_b0/<run_id>/` vía `modal volume ls` y que `make modal-pull`
  lo trae a `./outputs-remote`.
- **Regresión local:** correr `make train-baselines` en local sin `OUTPUT_ROOT` definido y
  confirmar que sigue escribiendo en `PROJECT_ROOT/outputs` (backward-compat de `get_output_root`).
- **Idempotencia:** segundo `make modal-seed` no re-descarga; segundo training reutiliza splits.

## Fuera de alcance (YAGNI)

- Pipeline principal `scripts/pipeline/train.py`.
- Multi-GPU / entrenamiento distribuido.
- Reemplazar o modificar `scripts/vastai/`.
- Automatizar la creación del Secret o del Volume desde el Makefile (se hace una vez a mano).

## Archivos afectados

| Archivo | Cambio |
|---|---|
| `src/config.py` | `get_output_root` lee `OUTPUT_ROOT` (backward-compatible) |
| `scripts/pipeline/train_baselines.py` | `main(argv=None)` + `parse_args(argv)` |
| `scripts/modal/train.py` | **nuevo** — App, Image, Volumes, entrypoints |
| `Makefile` | targets `modal-seed`, `modal-train-baselines`, `modal-pull` |
| `pyproject.toml` | extra `modal` |
| `.env.example` | `OUTPUT_ROOT` opcional documentado |
| `docs/es/deployment/modal.md` | **nuevo** — guía espejo de vast-ai.md |
