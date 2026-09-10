# Inferencia sobre el corpus completo

La corrida `2026-08-27_updated_hf_dataset` procesó el corpus de enfermedades, plagas y deficiencias publicado en Hugging Face [@huggingface2026dataset]. El inventario encontró 33,437 archivos válidos: 29,857 JPG, 3,573 JPEG y siete PNG. No hubo imágenes ilegibles. Se registraron cuatro duplicados distribuidos en cuatro grupos; la inferencia los conservó y dejó la deduplicación posterior al pipeline del clasificador.

La ejecución usó CPU, Ultralytics 8.4.104, `imgsz=640`, el perfil `mask_black` y el checkpoint C-01 con hash `4f66456d…f9c5ec6f`. Inició con reutilización del inventario, se reanudó de forma compatible y terminó sin errores. `run_metadata.json` conserva ambos comandos, commit de dataset `e515ab2f1e4c5729f8447520f1a630cf14c532dc`, commit del proyecto, configuración completa y tiempos acumulados.

Table: Resumen global de la corrida masiva.

| Resultado | Imágenes | Porcentaje |
|---|---:|---:|
| Procesadas | 33,437 | 100.00% |
| `reliable` | 28,360 | 84.82% |
| `uncertain` | 1,057 | 3.16% |
| `failed` | 4,020 | 12.02% |
| Salida segmentada publicada | 28,360 | 84.82% |
| Fallback a imagen completa | 5,077 | 15.18% |
| Errores de ejecución | 0 | 0.00% |

Las 29,417 imágenes con máscara seleccionada incluyen `reliable` y `uncertain`. Su área relativa media fue 0.68550 y la mediana 0.71829; el percentil 5 fue 0.18519 y el 95, 0.99604. La confianza media fue 0.94065 y la mediana 0.97484. Estas distribuciones son amplias: el corpus combina hojas pequeñas, encuadres cerrados y escenas donde casi todo el marco es tejido foliar.

![Las máscaras confiables y las inciertas ocupan rangos amplios; el gate usa geometría además de área.](figures/mask_area_distribution.png)

La inferencia del modelo sumó 5,265.25 segundos, con media de 157.47 ms por imagen. El tiempo de pared completo fue 11,330.30 segundos, aproximadamente 3 horas 8 minutos 50 segundos, lo que equivale a 2.95 imágenes por segundo incluyendo lectura, escritura, overlays, thumbnails y manifiestos. La mediana individual fue 78.16 ms; el percentil 95, 727.08 ms; y el máximo, 2,928.42 ms. La cola sugiere heterogeneidad de resolución y costes de I/O, por lo que la media no debe convertirse en un SLA móvil.

![La latencia en CPU tiene una cola larga; el gráfico recorta únicamente el 1% superior para conservar legibilidad.](figures/inference_time_distribution.png)

## Artefactos auditables

Por cada imagen se guardaron visualización de propuestas, máscara PNG, overlay, salida segmentada cuando existía, fallback enlazado cuando correspondía, cuatro thumbnails y JSON estructurado. Los manifiestos CSV y JSONL reúnen estado, clase, fuente, hashes, geometría, score, tiempo, warnings y rutas relativas. La galería permite revisar casos sin reconstruir inferencia.

::: evidence Diferencia de checkpoints
Esta corrida no mide el candidato D-01. Su `run_metadata.json` demuestra que utilizó C-01. Ejecutar D-01 sobre el mismo inventario permitiría una comparación pareada, pero no forma parte de la evidencia actual.
:::

