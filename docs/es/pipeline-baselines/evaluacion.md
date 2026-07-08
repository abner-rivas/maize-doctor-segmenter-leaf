# Evaluación

Cada baseline se evalúa sobre `test.csv` (N = 1 503, 9 clases) con transformaciones
deterministas. La métrica primaria es **macro-F1** (pondera por igual todas las clases, incluidas
las minoritarias); accuracy es secundaria por el desbalance.

## Resultados (test, 9 clases)

| Modelo | Accuracy | macro-F1 |
|---|---:|---:|
| `efficientnet_b0` | 0.9521 | **0.9146** |
| `shufflenet_v2_x1_0` | 0.9508 | 0.9030 |
| `efficientnet_lite0` | 0.9474 | 0.8951 |

`efficientnet_b0` lidera el macro-F1 y es el umbral de referencia actual.

## F1 por clase

| Clase | b0 | shufflenet | lite0 |
|---|---:|---:|---:|
| common_rust | 0.99 | 0.99 | 0.98 |
| fall_armyworm | 0.96 | 0.96 | 0.95 |
| gray_leaf_spot | 0.95 | 0.95 | 0.95 |
| healthy | 0.96 | 0.96 | 0.97 |
| lethal_necrosis | 0.99 | 0.99 | 1.00 |
| nitrogen_deficiency | 0.87 | 0.88 | 0.85 |
| northern_corn_leaf_blight | 0.95 | 0.94 | 0.94 |
| phosphorus_deficiency | 0.95 | 0.92 | 0.92 |
| **potassium_deficiency** | **0.62** | **0.54** | **0.49** |

## Hallazgos

- **`potassium_deficiency` es el cuello de botella universal** (266 imágenes totales, 40 en test):
  recall muy bajo en todos los modelos y responsable de la mayor caída del macro-F1.
- **Clúster de deficiencias N/P/K:** el potasio se confunde sobre todo con nitrógeno (en
  `efficientnet_b0`: de 40 imágenes de potasio, 21 correctas, 13→nitrógeno, 5→fósforo). Las tres
  deficiencias comparten síntomas de clorosis y son difíciles de separar con pocos ejemplos.
- **Clúster de lesiones foliares:** `gray_leaf_spot` ↔ `northern_corn_leaf_blight` se confunden
  mutuamente (lesiones visualmente parecidas).
- **Clases fáciles:** `common_rust`, `lethal_necrosis` y `healthy` (f1 ≥ 0.96 en todos los modelos).

La palanca de mayor impacto para el modelo final es aumentar/rebalancear las deficiencias
(especialmente potasio) antes de subir capacidad de modelo.
