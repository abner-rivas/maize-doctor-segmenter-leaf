# Fuentes y auditoría exploratoria

El dataset del segmentador se construyó desde dos exportaciones externas que conservaban anotaciones YOLO y respaldo COCO. La primera, `corn_leaf_diseases_classification`, incluye máscaras de hoja y de lesiones; la segunda, `corn`, contiene una clase de hoja más simple. Ambas declaran licencia CC BY 4.0 en los metadatos archivados por el proyecto. COCO se utilizó como respaldo para contrastar imágenes, categorías y polígonos por instancia [@lin2014coco].

Table: Inventario inicial de las fuentes externas.

| Fuente | Imágenes | Líneas YOLO | Polígonos válidos | Rol aceptado | Decisión |
|---|---:|---:|---:|---|---|
| `corn_leaf_diseases_classification` | 1,003 | 14,415 | 14,395 | Sólo clase `leaf` | Aceptar con filtrado |
| `corn` | 157 | 204 | 204 | Clase `leaf` | Aceptar con filtrado |

![El inventario distingue imágenes y etiquetas válidas antes de consolidar.](figures/eda_inventory.png)

La auditoría no asumió que toda anotación de segmentación representaba una hoja. En la fuente grande identificó 11,301 líneas de `gray_leaf_spot`, 2,091 de `northern_leaf_blight` y 1,023 de `leaf`. Las dos primeras son lesiones internas y se excluyeron. Se detectaron 20 líneas inválidas: 11 eran cajas YOLO mezcladas dentro del export de segmentación, ocho eran polígonos autointersectados y una tenía un vértice repetido. La fuente pequeña presentaba un TXT vacío respaldado por una imagen COCO sin anotaciones.

La correspondencia YOLO–COCO permitió recuperar una única anotación de hoja expresada como bbox. La recuperación exigía coincidencia única de imagen, clase, rol semántico, caja equivalente con tolerancia `1e-5`, polígono COCO válido y ausencia de alternativas. Las cajas correspondientes a lesiones se descartaron. No se reordenaron puntos ni se “repararon” polígonos ambiguos por intuición.

![Las áreas por clase confirman que las lesiones pequeñas no deben mezclarse con la máscara de hoja completa.](figures/eda_polygon_area.png)

La geometría refuerza la necesidad de separar fuentes. La mediana de área relativa de `leaf` fue 0.460459 en la fuente grande y 0.280577 en `corn`. Además, 71.57% de los polígonos válidos de `corn` tocaban el borde, frente a 9.07% de todos los polígonos válidos de la fuente grande; esta última cifra está dominada por lesiones y debe interpretarse por clase. El resultado no justifica eliminar recortes de borde, porque pueden ser ejemplos reales importantes.

![La fuente pequeña presenta una proporción mucho mayor de máscaras en contacto con el borde.](figures/eda_border_touching.png)

Se calcularon hashes SHA-256 para 1,160 imágenes candidatas y las 100 del piloto. No aparecieron duplicados exactos internos, cruzados o contra el piloto. El fingerprint global del EDA cubría 2,428 archivos y quedó registrado como `033db15c50a4ff2a23e5b152359dfb879b3c21b95a2f0bdd2742928e799a95ef`. Esta capa de procedencia permite invalidar cachés si cambia cualquier imagen, TXT, COCO, configuración o versión del parser.

