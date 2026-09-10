# Alcance y método de auditoría

## Corte temporal y repositorio

La auditoría corresponde al commit `985a920c5f3028ee81a7e0f7cd32507fa9b337c5` de la rama `master`, fechado el 17 de agosto de 2026. La corrida completa conserva el mismo commit en `run_metadata.json`, fue iniciada el 27 de agosto y reanudada hasta finalizar el 28 de agosto. La exportación para el clasificador registra fecha del 8 de septiembre. El checkout estaba `dirty`: faltaba el notebook de EDA y estaban sin seguimiento los módulos y pruebas de inferencia masiva. El informe trata esos archivos como parte del estado inspeccionado, pero no los confunde con contenido comprometido en Git.

## Fuentes de evidencia

Table: Jerarquía de evidencia utilizada en el informe.

| Nivel | Evidencia | Uso | Restricción |
|---|---|---|---|
| Primario | `summary.json`, `run_metadata.json`, `manifest.csv`, checkpoint C-01 | Corrida masiva, configuración, hashes y tiempos | No contiene ground truth de máscaras |
| Primario documentado | Resumen D-01, `results.csv` y evaluación downstream citados por la documentación | Métricas de entrenamiento y validación | Los archivos de output no están en el checkout actual |
| Canónico documentado | Locks, manifiestos y splits bajo `data/leaf_detection` | Composición y prevención de fugas | El árbol `data/leaf_detection` está vacío en esta copia |
| Código | Grafo Codebase Memory y snippets de `src/` | Arquitectura, contratos, fallbacks y validaciones | Describe comportamiento implementado, no resultados empíricos |
| Pruebas | `pytest` y `ruff` ejecutados en el entorno local | Estado del software | Cuatro pruebas requieren datos canónicos ausentes |

El repositorio fue indexado con Codebase Memory en modo completo. El grafo obtenido contiene 2,271 nodos y 9,964 relaciones, con 764 funciones, 331 métodos, 103 clases y 160 archivos. Se utilizaron búsquedas estructurales para ubicar símbolos y trazas, y lectura directa sólo para documentos, YAML, JSON, CSV y otros artefactos no cubiertos por el grafo. Esta estrategia reduce el riesgo de inferir arquitectura únicamente a partir de nombres de carpetas.

## Límites del análisis

No se entrenó ni reanudó un modelo, no se abrió el test interno, no se ejecutó el piloto externo y no se modificaron thresholds. Tampoco se reconstruyeron las fuentes externas o los splits ausentes. Esas acciones están protegidas por guards explícitos y algunas implican coste de GPU o consumo irreversible de conjuntos retenidos. El informe analiza la evidencia disponible sin ampliar esa autorización.

La corrida masiva sí permite análisis estadístico nuevo porque su manifiesto ya existe. Se calcularon tasas por clase, cuantiles de confianza, área y latencia sin cambiar los datos. Los gráficos se generan determinísticamente desde las 33,437 filas. Para evitar selección visual oportunista, el panel cualitativo utiliza identificadores fijados por reglas documentadas: un caso `reliable` cercano a la mediana de confianza y casos representativos de geometría sospechosa y ausencia de detección.

::: risk Evidencia ausente
Las cifras históricas de consolidación y D-01 están coherentemente repetidas en varios documentos, configuraciones y hashes, pero no fue posible recalcularlas desde los datos originales en este checkout. El informe conserva esa diferencia entre “registrado por el proyecto” y “reverificado ahora”.
:::

