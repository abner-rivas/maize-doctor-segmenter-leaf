# Objetivos

## Objetivo general

Documentar de forma reproducible el diseño, los datos, el entrenamiento, la evaluación y la integración del segmentador de hojas de Doctor Maíz, distinguiendo la evidencia experimental con máscaras de referencia de la evidencia operativa producida sobre el corpus completo de clasificación.

## Objetivos específicos

1. Reconstruir la arquitectura real desde el grafo de código, las funciones centrales y los puntos de entrada ejecutables.
2. Registrar la procedencia y las reglas con que dos fuentes YOLO/COCO se filtraron hasta obtener una clase única de hoja.
3. Explicar el cierre humano del dataset, su fingerprint y la creación de splits agrupados sin fugas conocidas.
4. Describir los gates locales y cloud que protegen dependencias, pesos, GPU, smoke, entrenamiento, reanudación y evaluación final.
5. Comparar C-01 con D-01 bajo el cambio controlado `mosaic=0.0`, sin presentar validación como test final.
6. Formalizar la conversión de propuestas Ultralytics, el score determinista de selección y los perfiles de salida.
7. Evaluar el quality gate, sus thresholds, su calibración humana y las causas de `uncertain` o `failed`.
8. Reportar IoU, Dice, recall, precisión, subsegmentación y sobresegmentación sobre el conjunto con verdad de referencia disponible.
9. Caracterizar la corrida de 33,437 imágenes por clase, estado, causa de rechazo, área de máscara y latencia.
10. Verificar la exportación del subconjunto confiable al formato esperado por el clasificador y sus efectos sobre el balance.
11. Auditar reproducibilidad mediante commits, hashes, manifiestos, comandos y políticas de no sobrescritura.
12. Revisar el estado actual de pruebas, lint y disponibilidad local de datos sin modificar los cambios del usuario.
13. Identificar riesgos técnicos, sesgos de selección, límites de generalización y salvaguardas para uso agrícola responsable.
14. Proponer una secuencia de cierre que preserve test y piloto hasta congelar definitivamente configuración y semillas.

::: evidence Criterio de éxito del informe
Cada cifra central debe poder rastrearse a un JSON, CSV, YAML, checkpoint con SHA-256 o documento técnico del repositorio. Las inferencias analíticas, como atribuir una caída de cobertura a un posible cambio de dominio, se presentan como hipótesis y no como hechos demostrados.
:::

