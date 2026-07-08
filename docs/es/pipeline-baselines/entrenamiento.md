# Entrenamiento

Los baselines se entrenan con `scripts/pipeline/train_baselines.py`, que comparte toda la
infraestructura de datos y modelos con el pipeline principal. Cada arquitectura se construye
desde `MODEL_REGISTRY`, con backbone pre-entrenado en ImageNet y solo la capa de clasificación
reemplazada por una lineal de 9 salidas.

## Configuración canónica

| Hiperparámetro | Valor |
|---|---|
| Clases | 9 (perfil `baseline`, cap 1 500/clase) |
| Épocas | 30 |
| Optimizador | AdamW (lr 1e-4, weight decay 1e-4) |
| Pérdida | CrossEntropyLoss ponderada por clase |
| Balanceo | WeightedRandomSampler + augmentation minority en caliente |
| Pesos iniciales | ImageNet (`pretrained=True`) |
| `image_size` | por-modelo (los 3 baselines usan 224×224) |
| Batch size | 32 (auto-escalado según resolución del modelo) |

Cada corrida se versiona en `outputs/baselines/<modelo>/<run_id>/` con `best.pth`, `summary.json`,
`predictions.csv`, `train_history.csv` y los reportes de test. El `summary.json` es la fuente de
verdad del mapeo clase→índice y del `image_size` con que se entrenó el checkpoint.

## Modelos baseline

Se adoptan tres arquitecturas ligeras que cubren el eje precisión↔eficiencia y convierten a
TensorFlow Lite para despliegue móvil offline:

- `efficientnet_b0`
- `shufflenet_v2_x1_0`
- `efficientnet_lite0`

## Ejecución

```bash
# Local
make train-baselines MODELS=all EPOCHS=30

# GPU en Modal (cap 1500 sobre 9 clases; regenera splits si cambió el perfil)
make modal-train-baselines MODELS=all EPOCHS=30 MAX_PER_CLASS=1500 REGEN_SPLITS=1
```
