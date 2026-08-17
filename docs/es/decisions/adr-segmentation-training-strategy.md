# ADR: estrategia de entrenamiento del segmentador

## Estado

Aceptado.

## Decisión

Entrenar `yolo26n-seg` con un dataset congelado de clase única y splits agrupados
70/15/15. La prioridad es preservar tejido de hoja y limitar fondo residual; por ello las
métricas downstream incluyen IoU, Dice, recall de máscara, precisión y área extra.

El flujo exige:

1. locks y fingerprints válidos;
2. preflight sin mutar el dataset;
3. smoke autorizado en GPU;
4. configuración final congelada;
5. entrenamiento/reanudación con confirmación literal;
6. test interno una sola vez;
7. piloto externo sólo después de aprobar el test.

Ultralytics queda fijado a `8.4.104` para que el checkpoint y el parser de etiquetas
mantengan el mismo contrato.
