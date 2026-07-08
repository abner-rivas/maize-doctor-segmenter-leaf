# Evaluación

Cada baseline se evalúa sobre `test.csv` (N = 1 503, 9 clases) con transformaciones
deterministas. La métrica primaria es **macro-F1** (pondera por igual todas las clases, incluidas
las minoritarias); accuracy es secundaria por el desbalance.

## Resultados (test, 9 clases)

Estos son los números que deciden cuál arquitectura se adopta como referencia: la tabla compara
accuracy y macro-F1 de los tres baselines sobre el conjunto de test.

| Modelo | Accuracy | macro-F1 |
|---|---:|---:|
| `efficientnet_b0` | 0.9521 | **0.9146** |
| `shufflenet_v2_x1_0` | 0.9508 | 0.9030 |
| `efficientnet_lite0` | 0.9474 | 0.8951 |

`efficientnet_b0` lidera el macro-F1 y es el umbral de referencia actual.

## F1 por clase

La cifra agregada esconde que el rendimiento no es parejo entre clases. Esta tabla desglosa el F1
por clase y por modelo, y ya deja ver dónde se concentra el problema.

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

Cruzando ambas tablas aparecen patrones consistentes en los tres modelos, que conviene tener en
cuenta antes de decidir el siguiente paso.

**`potassium_deficiency` es el cuello de botella universal.** Con apenas 266 imágenes totales (40
en test), el recall es muy bajo en los tres modelos y es la clase que más castiga el macro-F1. El
problema no aparece aislado: forma parte de un clúster de deficiencias N/P/K en el que el potasio
se confunde sobre todo con nitrógeno (en `efficientnet_b0`, de 40 imágenes de potasio, 21 se
clasifican correctamente, 13 se confunden con nitrógeno y 5 con fósforo). Las tres deficiencias
comparten síntomas de clorosis y son difíciles de separar con pocos ejemplos.

Hay un segundo clúster de confusión, esta vez entre lesiones foliares: `gray_leaf_spot` y
`northern_corn_leaf_blight` se confunden mutuamente porque sus lesiones son visualmente
parecidas. En el otro extremo están las clases fáciles, `common_rust`, `lethal_necrosis` y
`healthy`, que mantienen un F1 igual o superior a 0.96 en los tres modelos.

La palanca de mayor impacto para el modelo final es aumentar o rebalancear las deficiencias
(especialmente potasio) antes de subir capacidad de modelo.
