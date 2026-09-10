# Calidad de software y verificación actual

El índice estructural muestra una base de código amplia para el tamaño del modelo: 764 funciones, 331 métodos y 103 clases en Python, YAML, Bash, TypeScript, Vue, TOML y CSS. Los clústeres principales corresponden a procesamiento, inferencia masiva, consolidación, splits, revisión visual, control Modal y evaluación. Las fronteras más relevantes conectan `segmentation` con `preprocessing`, `experiments` con `evaluation` y `experiments` con `preprocessing`.

Las funciones centrales tienen responsabilidades acotadas. `UltralyticsLeafSegmenter.segment` sólo ejecuta y convierte inferencia; `select_target_leaf` puntúa y traza; `SegmentedLeafProcessor.process` materializa perfiles; `assess_segmentation` asigna estado; `_process_one` persiste artefactos; `image_metrics` calcula comparación por píxel; y `export_classifier_dataset` selecciona y publica de forma atómica. Este encadenamiento facilita probar cada política sin cargar un modelo real.

## Pruebas ejecutadas

La suite completa se ejecutó con Python 3.12.3, sin caché de pytest y sin escribir bytecode. El resultado fue 349 pruebas aprobadas, cuatro fallidas y tres subtests aprobados en 19.05 segundos. Las cuatro fallas comparten una causa externa: el checkout no contiene `data/leaf_detection/detector_dataset` ni los 211 archivos del piloto que esperan las pruebas de empaquetado.

Table: Resultado de verificación del checkout inspeccionado.

| Comprobación | Resultado | Interpretación |
|---|---|---|
| `ruff check` | Aprobado | Sin hallazgos del linter |
| Pytest total | 349 pass, 4 fail | La lógica aislada aprueba; faltan datos canónicos |
| Test de status read-only | Falla por `dataset_lock.json` ausente | No observó una mutación |
| Verificación de payload cloud | Falla por dataset ausente | No recalculó fingerprints |
| Allow-list del paquete | Falla por `images/train` ausente | No llegó a evaluar inclusión |
| Manifiesto de transporte del piloto | Esperaba 211, obtuvo 0 | El piloto no está materializado |

No se corrigieron estas pruebas creando datos ficticios o debilitando asserts. Restaurar los artefactos canónicos es la solución apropiada. Las 349 aprobadas cubren, entre otros, geometría, máscaras, letterbox, selección, quality gate, calibración, inferencia por dataset, exportación, consolidación, splits, preflight, control cloud y seguridad del Makefile.

## Complejidad y deuda visible

El flujo actual documenta pasadas repetidas sobre hashes y datos: los splits se materializan tres veces para probar determinismo; `cloud-prepare` recorre el payload en cuatro etapas; y el gate cloud verifica el paquete dos veces por corrida. Algunas duplicaciones son defensivas, pero elevan I/O sobre 2.3 GB. La comparación perceptual en `build_split_groups` es cuadrática sobre 1,155 elementos; es aceptable en el tamaño actual, pero necesitará indexación si crece sustancialmente.

También existe una dependencia rígida del nombre `yolo26n_seg_baseline` en scripts de validación. Una segunda corrida que Ultralytics renombre con sufijo podría dejar consumidores apuntando al primer directorio. D-01 evita ese choque con nombres propios, pero conviene que todos los consumidores resuelvan checkpoint por manifiesto y hash, no por ruta fija.

::: risk Estado Git
El pipeline de inferencia masiva y sus pruebas están sin seguimiento en Git, y el notebook de EDA aparece eliminado. El reporte documenta el comportamiento encontrado, pero esos cambios deben revisarse y confirmarse antes de considerar reproducible el checkout desde un clone limpio.
:::

