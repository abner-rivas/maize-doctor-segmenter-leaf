# Splits y prevención de fugas

La división utiliza semilla 42 y proporciones objetivo 70/15/15. No asigna archivos independientemente. Primero construye componentes conexos con cinco señales: SHA-256 exacto, grupo de duplicado, grupo de variante Roboflow, combinación de fuente y nombre original, y hash perceptual promedio de 64 bits con distancia Hamming máxima de cuatro. Si dos imágenes se relacionan por cualquiera de esas señales, sus componentes se unen y permanecen en un mismo split.

La implementación `build_split_groups` ordena por nombre, aplica union-find para coincidencias exactas y compara hashes perceptuales por pares. Las 1,155 imágenes forman 1,035 grupos, con tamaño máximo 12. `assign_groups_to_splits` ordena primero grupos grandes y raros, minimiza un costo multivariado y penaliza el desbordamiento de los conteos objetivo. El balance considera imágenes, máscaras, fuente, orientación, resolución y bins de área; un refinamiento posterior intercambia grupos del mismo tamaño cuando reduce el costo.

Table: Resultado de los splits congelados.

| Split | Imágenes | Proporción real | Máscaras | Grupos | Fuente `corn` | Fuente grande |
|---|---:|---:|---:|---:|---:|---:|
| Train | 809 | 70.0433% | 858 | 724 | 109 | 700 |
| Validación | 173 | 14.9784% | 183 | 156 | 23 | 150 |
| Test | 173 | 14.9784% | 183 | 155 | 23 | 150 |

![Los conteos de imágenes y máscaras conservan las proporciones objetivo sin dividir grupos relacionados.](figures/dataset_splits.png)

![La procedencia de las dos fuentes queda representada en los tres splits.](figures/source_splits.png)

Los fingerprints son `06035eed…d29ba` para train, `3c7bf7ab…4e720` para validación, `04654535…89c51` para test y `96833e43…0c0e1` para la asignación combinada. El validador registró cero hashes, grupos, variantes Roboflow o vecinos perceptuales compartidos entre splits; también cero cruces exactos, nominales, por base o perceptuales con el piloto.

La generación se repitió dos veces en directorios temporales y produjo asignaciones y fingerprints idénticos. La reconstrucción posterior tomó `filename → split` como contrato y verificó cero cambios en los 1,155 nombres. Los JPEG canónicos de `all/images` y los tres splits pasaron marcador EOI, `Image.verify()`, carga completa y el checker real de Ultralytics 8.4.104 sin cambios de hash.

## Test interno y piloto externo

El test de 173 imágenes pertenece a las mismas dos fuentes consolidadas, pero permanece retenido para una sola evaluación después de congelar configuración y semillas. El piloto de 100 imágenes es externo a train, validación y test: sirve únicamente como auditoría cualitativa final. Esta separación evita usar el conjunto más independiente para seleccionar thresholds o corregir el modelo.

::: risk Estado actual del checkout
Los manifiestos, locks e imágenes descritos en esta sección no están materializados bajo `data/leaf_detection` en la copia inspeccionada. Sus hashes y conteos sobreviven en documentación y en el manifiesto del paquete cloud, pero la verificación local completa requiere restaurar los datos canónicos.
:::

