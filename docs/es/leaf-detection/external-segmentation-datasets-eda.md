# Auditoría de datasets externos de segmentación

## Alcance

Esta auditoría evalúa dos fuentes externas como candidatas para un futuro
segmentador de hojas de maíz. No entrenó YOLO, no descargó pesos, no reparó
etiquetas, no consolidó datasets y no modificó clasificadores ni checkpoints.

El notebook reproducible es
[`notebooks/02_leaf_segmentation_external_sources_eda.ipynb`](../../../notebooks/02_leaf_segmentation_external_sources_eda.ipynb).
La lógica reutilizable está en `src/data/segmentation_audit.py` y los resultados
completos en `outputs/leaf_detection/external_sources_eda/`.

La motivación histórica está en [Historia del aislamiento de hojas](history.md)
y la decisión formal sobre estas fuentes en
[ADR: datasets externos](../decisions/adr-external-leaf-segmentation-datasets.md).

YOLO tendrá una sola responsabilidad: segmentar hojas. No clasificará
enfermedades. Un clasificador posterior deberá entrenarse con el mismo
preprocesamiento de segmentación usado en evaluación e inferencia.

## Fuentes, trazabilidad y licencia

| Nombre lógico | YOLO | Respaldo COCO | Licencia declarada |
|---|---|---|---|
| `corn_leaf_diseases_classification` | `data/leaf_detection/external_sources/corn_leaf_diseases_classification_yolo26/` | `data/leaf_detection/external_sources/corn_leaf_diseases_classification_coco_segmentation/` | CC BY 4.0 |
| `corn` | `data/leaf_detection/external_sources/corn_yolo26/` | `data/leaf_detection/external_sources/corn_coco_segmentation/` | CC BY 4.0 |

Se leyeron `data.yaml`, `README.dataset.txt`, `README.roboflow.txt` y las
categorías COCO. La auditoría registra hashes SHA-256 de esos metadatos y del
JSON COCO, además de hashes por imagen. La licencia debe conservar su
atribución cuando se materialice un dataset derivado.

El piloto de 100 imágenes en `data/leaf_detection/pilot/images/` fue usado sólo
para comprobar duplicados. Continúa retenido como test y no forma parte de las
fuentes candidatas para entrenamiento.

## Metodología

La auditoría:

1. inventarió imágenes, TXT, metadatos y anotaciones COCO;
2. comprobó la correspondencia imagen-etiqueta;
3. distinguió bbox YOLO de polígonos y validó sintaxis y topología;
4. calculó estadísticas de imágenes, polígonos e instancias;
5. contrastó YOLO con COCO por nombre de imagen y clase;
6. calculó duplicados exactos SHA-256 internos, cruzados y contra el piloto;
7. generó muestras visuales deterministas con semilla 42;
8. produjo una muestra estratificada para revisión manual.

La caché usa esquema 2 y parser 2. Su fingerprint global se construye con rutas
relativas ordenadas y SHA-256 individuales de imágenes, TXT, JSON COCO,
`data.yaml`, metadatos, configuración del análisis y las 100 imágenes del
piloto. El resultado actual cubre 2 428 archivos y tiene SHA-256 global
`033db15c50a4ff2a23e5b152359dfb879b3c21b95a2f0bdd2742928e799a95ef`.
Cambiar cualquier TXT, imagen, archivo del piloto o versión del parser invalida
la caché.

El área de los polígonos se calculó con la fórmula del zapatero. Los umbrales
`< 0.05`, `0.05–0.50` y `> 0.50` sólo se usaron como indicadores exploratorios
de máscaras pequeñas, medianas y grandes.

La comprobación opcional contra las 31,622 imágenes del dataset completo quedó
desactivada mediante `RUN_FULL_DATASET_DUPLICATE_CHECK = False`.

## Inventario y clases

| Fuente | Imágenes | TXT | Líneas YOLO | Válidas | Inválidas | TXT vacíos | Sin TXT | TXT sin imagen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `corn_leaf_diseases_classification` | 1,003 | 1,003 | 14,415 | 14,395 | 20 | 0 | 0 | 0 |
| `corn` | 157 | 157 | 204 | 204 | 0 | 1 | 0 | 0 |

![Inventario de imágenes y etiquetas](/leaf_detection/external_sources_eda/inventory_counts.png)

| Fuente | ID YOLO | Clase | Rol semántico | Líneas | Polígonos válidos | Imágenes | Decisión |
|---|---:|---|---|---:|---:|---:|---|
| `corn_leaf_diseases_classification` | 0 | `gray_leaf_spot` | lesión | 11,301 | 11,293 | 500 | excluir |
| `corn_leaf_diseases_classification` | 1 | `leaf` | hoja completa | 1,023 | 1,021 | 1,001 totales; 1,000 válidas | incluir tras remapeo |
| `corn_leaf_diseases_classification` | 2 | `northern_leaf_blight` | lesión | 2,091 | 2,081 | 500 | excluir |
| `corn` | 0 | `leaf` | hoja completa | 204 | 204 | 156 | incluir tras remapeo |

![Polígonos por clase](/leaf_detection/external_sources_eda/class_polygon_counts.png)

La inspección visual confirma que `leaf` representa el contorno de la hoja
completa en ambas fuentes. `gray_leaf_spot` y `northern_leaf_blight` representan
lesiones dentro de la hoja; no son clases válidas para un segmentador binario
de hoja.

## Integridad sintáctica y topológica

La lectura directa no encontró coordenadas concatenadas, tokens no numéricos,
NaN, infinitos, pares incompletos ni coordenadas fuera de `[0,1]`.

En la fuente grande hay 20 líneas inválidas. Once son filas de cinco campos en
formato YOLO de detección `class x_center y_center width height`, clasificadas
como `bbox_format_in_segmentation_label`; no son polígonos truncados:

- 7 pertenecen a `gray_leaf_spot`;
- 1 pertenece a `leaf`;
- 3 pertenecen a `northern_leaf_blight`.

La validación topológica detectó además ocho polígonos autointersectados —seis
`northern_leaf_blight`, uno `gray_leaf_spot` y uno `leaf`— y otro polígono NLB
con vértice repetido. Se registran `self_intersection`, `repeated_vertex` y
`non_simple_polygon`; también se comprueban aristas nulas, vértices únicos
insuficientes y área nula o casi nula. No se reordenaron puntos ni se cambió
ningún TXT.

La fuente `corn` contiene una imagen con un TXT vacío:
`20250809_184623_jpg.rf.xkD9slgUmnNC0bOvP8pn.jpg`. Su respaldo COCO también
tiene cero anotaciones, por lo que no puede recuperarse desde COCO y debe
excluirse o revisarse manualmente.

## Geometría e instancias

| Fuente/clase | N válido | Área media | Área mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|
| grande / `gray_leaf_spot` | 11,293 | 0.001120 | 0.000563 | 0.0000003 | 0.051089 |
| grande / `leaf` | 1,021 | 0.473211 | 0.460459 | 0.000085 | 0.997224 |
| grande / `northern_leaf_blight` | 2,081 | 0.011370 | 0.006075 | 0.000041 | 0.849409 |
| `corn` / `leaf` | 204 | 0.287812 | 0.280577 | 0.011031 | 0.791384 |

La mediana global de la fuente grande es `0.000821`, pero está dominada por
lesiones pequeñas y no representa el área de la hoja completa. Las estadísticas
por clase son las que deben guiar una consolidación futura.

![Área relativa por clase](/leaf_detection/external_sources_eda/polygon_area_by_class.png)

| Fuente | Media de instancias | Mediana | Máximo | 0 polígonos | 1 | 2 | 3 o más |
|---|---:|---:|---:|---:|---:|---:|---:|
| `corn_leaf_diseases_classification` | 14.37 | 9 | 77 | 0 | 0 | 117 | 886 |
| `corn` | 1.30 | 1 | 5 | 1 | 125 | 18 | 13 |

La fuente grande tiene 1,000 imágenes con al menos una hoja completa
topológicamente válida y 1,000 con lesiones; normalmente mezcla una hoja y
múltiples lesiones. La fuente pequeña
tiene 156 imágenes con hojas válidas y ninguna clase de lesión.

El `9.07 %` de los polígonos válidos de la fuente grande toca al menos un borde,
frente al `71.57 %` en `corn`. Esta diferencia debe revisarse al diseñar
augmentations y criterios de recorte.

![Máscaras que tocan bordes](/leaf_detection/external_sources_eda/border_touching_percent.png)

## Contraste YOLO–COCO

Los respaldos COCO contienen 1,003 imágenes y 14,415 anotaciones para la fuente
grande, y 157 imágenes y 204 anotaciones para `corn`. No hay diferencias en
conteos de anotaciones por imagen ni en nombres de clases.

Las 11 filas bbox tienen una correspondencia COCO única y topológicamente
válida. La consolidación recuperó únicamente la fila `leaf`; las otras diez son
lesiones. La coincidencia no depende de la posición: exige imagen única, clase
y rol semántico compatibles, bbox equivalente con tolerancia `1e-5`, polígono
COCO válido y ausencia de alternativas. El error máximo observado fue
`3.2553e-6`; se registran error, IoU y criterios de compatibilidad.

Los nueve polígonos YOLO no simples no tienen una alternativa COCO
topológicamente válida, por lo que no se recuperan automáticamente.

La imagen con TXT vacío también tiene cero anotaciones en COCO y no es
recuperable con ese respaldo.

## Duplicados y fuga

Se calcularon hashes de las 1,160 imágenes candidatas y de las 100 imágenes del
piloto:

- duplicados internos en la fuente grande: 0 grupos;
- duplicados internos en `corn`: 0 grupos;
- duplicados exactos entre las dos fuentes: 0 grupos;
- duplicados contra el piloto: 0 grupos;
- fuga con el piloto: no detectada.

## Auditoría visual pendiente

Se generaron 207 archivos bajo
`outputs/leaf_detection/external_sources_eda/previews/`, organizados en:

- muestras aleatorias;
- ejemplos separados por clase;
- máscaras pequeñas y grandes;
- máscaras con más puntos;
- múltiples instancias;
- máscaras que tocan bordes;
- etiquetas sospechosas.

No se generaron previews de duplicados cruzados porque no existen casos. El CSV
`manual_semantic_review.csv` contiene 32 casos estratificados; los campos
`annotation_quality`, `multiple_leaves` y `background_complexity` permanecen
como `unknown` hasta completar una revisión humana.

## Decisión

| Fuente | Estado | Candidatas con hoja válida | Condiciones |
|---|---|---:|---|
| `corn_leaf_diseases_classification` | `accepted_with_filtering` | 1,000 | conservar sólo `leaf` topológicamente válida, recuperar el bbox de hoja, revisar la hoja autointersectada, excluir lesiones y remapear |
| `corn` | `accepted_with_filtering` | 156 | excluir o revisar la imagen vacía, revisar recortes de borde y remapear |

Al cierre del EDA, las fuentes **todavía no estaban listas para
consolidarse**. En ese momento se definieron estos pasos:

1. completar la revisión visual estratificada;
2. definir reglas de filtrado sin aplicarlas a las fuentes originales;
3. decidir y documentar la recuperación desde COCO;
4. materializar un dataset derivado con `0 = maize_leaf`;
5. crear splits libres de duplicados y mantener el piloto sólo como test;
6. validar el dataset consolidado en una tarea separada.

La consolidación auditada era el siguiente experimento, no el entrenamiento.
Ese paso ya se completó, como documenta la sección siguiente. Sólo después de
aprobar el pool se crearán splits propios del segmentador, se confirmará la
arquitectura y se evaluará contra el piloto retenido.

Estos datos pueden ayudar a mejorar el aislamiento de la hoja, pero no prueban
una mejora de precisión del clasificador. El segmentador deberá evaluarse y el
clasificador deberá entrenarse después con las imágenes procesadas. No se debe
activar segmentación sólo durante inferencia sobre checkpoints entrenados con
imágenes completas.

## Resultado de la consolidación posterior

La fase siguiente materializó un pool candidato bajo
`data/leaf_detection/detector_dataset/all/`:

- 1 160 imágenes consideradas y 1 156 incluidas;
- 1 226 polígonos finales, todos remapeados a `0 = maize_leaf`;
- 13 392 anotaciones de lesión excluidas;
- una anotación `leaf` recuperada desde COCO;
- cero duplicados exactos eliminados y cero cruces contra el piloto;
- 34 filas de revisión manual: 32 de la muestra estratificada, un TXT vacío y
  una hoja autointersectada excluida del pool;
- dos casos visuales obligatorios con estado pendiente: la hoja
  autointersectada y la recuperación COCO de área `4.2767e-7`.

Las validaciones estructurales pasaron sin errores. Aún no se crearon splits:
los previews y la cola manual deben aprobarse antes de usar el pool para
entrenamiento. Los reportes viven en
`outputs/leaf_detection/detector_dataset_consolidation/`.

## Artefactos

- `summary.json`: resultado estructurado y controles de seguridad;
- `source_summary.csv`, `class_summary.csv`, `class_instance_summary.csv`,
  `source_comparison.csv`;
- `image_statistics.csv`, `polygon_statistics.csv`;
- `annotation_issues.csv`, `image_label_mismatches.csv`;
- `yolo_coco_comparison.csv`, `duplicate_report.csv`;
- `manual_semantic_review.csv`, `decision_summary.md`;
- `charts/` y `previews/`.

La configuración activa se mantiene sin cambios:

```yaml
processing_profile: baseline_full

leaf_detection:
  enabled: false
```
