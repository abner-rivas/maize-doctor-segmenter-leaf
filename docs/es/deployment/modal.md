# Entrenamiento en Modal

Guía para entrenar los **baselines** y correr **explicabilidad** en GPU de
[Modal](https://modal.com/docs/guide). Modal coexiste con vast.ai (ver `vast-ai.md`); no lo
reemplaza. A diferencia de vast.ai (VM + SSH), en Modal defines código que corre en la nube y
se cobra por segundo, con auto-teardown (no hay que acordarse de destruir instancias).

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

# 2) Entrenar baselines en GPU (A10 por defecto) — mismas banderas que make train-baselines
make modal-train-baselines MODELS=efficientnet_b0 EPOCHS=30
make modal-train-baselines MODELS=efficientnet_b0 NO_CAP=1          # sin tope de imágenes
make modal-train-baselines MODELS=efficientnet_b0 LIME=1            # + reportes LIME al terminar

# 3) Explicabilidad post-hoc sobre un run ya entrenado
make modal-explain-lime MODELS=efficientnet_b0
make modal-explain-report MODELS=efficientnet_b0 SAMPLE_SIZE=50
make modal-explain-errors MODELS=efficientnet_b0

# 4) Traer los resultados al equipo local
make modal-pull            # copia el volumen corn-outputs -> ./outputs-remote

# Limpiar el volumen de outputs (splits/runs/reportes) si hace falta empezar de cero
make modal-clean-outputs
```

`scripts/modal/train.py` expone la misma CLI que `train_baselines.py` (`--models`, `--epochs`,
`--no-cap`/`--max-per-class`, `--batch-size`, `--image-size`, `--learning-rate`,
`--weight-decay`, `--num-workers`, `--no-pretrained`, `--lime`) — cualquier combinación que
funcione en local funciona igual en Modal. Igual para `scripts/modal/explain.py` respecto a
`explain_lime.py`/`explain_report.py` (`--run`, `--baseline`, `--sample-size`, `--num-samples`,
`--errors-only`; `--image`/`--output` de `explain-lime` también, pero deben ser rutas dentro
del contenedor, relativas a `/data` u `/outputs`, no del filesystem local).

Equivalente sin `make`:
```bash
modal run scripts/modal/train.py::seed_dataset
modal run scripts/modal/train.py --models "efficientnet_b0" --epochs 30
modal run scripts/modal/explain.py::explain_report --models "efficientnet_b0" --sample-size 50
modal run scripts/modal/train.py::clean_outputs
modal volume get corn-outputs / ./outputs-remote
```

## Notas

- Los resultados se versionan por corrida en `/outputs/baselines/<modelo>/<run_id>/`
  (igual que en local; `run_id` = timestamp). `make modal-pull` los baja a `./outputs-remote`.
- Los splits baseline se generan la primera vez y se reutilizan (lazy) en corridas siguientes —
  mismo comportamiento que `train_baselines.py` en local. Si ya existen con otro tope de
  imágenes, no se regeneran solos: corre `make modal-clean-outputs` primero.
- Para cambiar la GPU, edita `gpu="A10"` en `scripts/modal/train.py`/`explain.py` (opciones:
  T4, L4, A10, L40S, A100, H100…).
- Solo cubre baselines y su explicabilidad; el pipeline principal (`train.py`) no está integrado.
