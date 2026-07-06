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
