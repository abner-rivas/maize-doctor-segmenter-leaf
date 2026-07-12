# Evaluación

La evaluación se hará sobre `test.csv` con transformaciones deterministas y **macro-F1** como métrica primaria (por el desbalance, con accuracy como secundaria). Se conservará el desglose de F1 por clase y la matriz de confusión, y se añadirán las métricas que se describen a continuación.

## Evaluación cross-environment

Para descartar que el modelo aprenda el fondo en vez de la lesión, se evaluará de forma cruzada por entorno: entrenar o medir en un dominio (laboratorio o campo) y comprobar el rendimiento en el otro. Una caída fuerte al cambiar de dominio sería evidencia de que el modelo se apoya en atajos de fondo, lectura que se cruzará con la [interpretabilidad](./interpretabilidad).

## Benchmark de despliegue

Como el destino es un dispositivo móvil, se medirá la viabilidad de despliegue: latencia de inferencia, throughput, tamaño del modelo exportado (por ejemplo a TensorFlow Lite) y, en lo posible, el comportamiento en el hardware objetivo.
