# Consolidación y revisión humana

La consolidación materializó un pool derivado sin tocar las fuentes originales. Consideró 1,160 imágenes, excluyó casos sin hoja aceptable y conservó inicialmente 1,156 candidatas con 1,226 polígonos remapeados a `0 = maize_leaf`. En total se retiraron 13,392 anotaciones de lesión. Cada fila mantuvo nombre original, fuente, hash, licencia, clase previa, método de recuperación, estado de calidad y grupo de variante.

El gate no cerró el dataset automáticamente. Se creó una cola humana de 35 casos únicos: 32 de una muestra visual estratificada, un TXT vacío, una hoja autointersectada y la anotación COCO recuperada extremadamente pequeña, con solapamientos entre listas. Los previews resolvieron geometría desde YOLO, COCO, el manifiesto de recuperación y finalmente el consolidado. El registro final documenta 16 decisiones `approved`, 16 `exclude` y tres casos que debían reanotarse; quedaron cero pendientes y cero contradicciones.

Después de aplicar esas decisiones, el padre congelado contiene 1,155 imágenes, 1,155 TXT y 1,224 máscaras. Una imagen JPEG era decodificable pero carecía de marcador EOI. La copia derivada añadió `FF D9` sin recodificar: cambió el hash binario, pero mantuvo el hash de píxeles RGB `d63697cd3d48aef376302d841baaae2a666fd200d39c05955a8173ee740f70c6`. Esta corrección quedó registrada en `image_normalization_manifest.csv` y no alteró identidad lógica ni asignación.

Table: Resultado del cierre del dataset padre.

| Control | Resultado |
|---|---:|
| Imágenes finales | 1,155 |
| Archivos TXT | 1,155 |
| Máscaras `maize_leaf` | 1,224 |
| Clases finales | 1 |
| Anotaciones de lesión excluidas | 13,392 |
| Casos humanos únicos | 35 |
| Casos pendientes al cierre | 0 |
| Duplicados exactos detectados | 0 |
| Cruces exactos con piloto | 0 |

El fingerprint del padre final es `7a4a5c083fc64b067df12bcc95ec976d5a7e3b8a585d0a090b6b3940af4d7d5c`. El lock cambia a `ready_for_split_generation` sólo cuando conteos, decisiones y hashes concuerdan. La función `validate_consolidated_dataset` reabre imágenes, vuelve a parsear polígonos y comprueba clase, coordenadas, topología y correspondencia. El valor del lock no es decorativo: los pasos de split y preflight lo recalculan antes de continuar.

::: evidence Intervención humana acotada
Las decisiones humanas no se infirieron desde scores del modelo. Se mantuvieron como un insumo explícito e irreversible del cierre. El pipeline puede regenerar previews y reportes, pero no inventar ni sustituir el juicio del revisor.
:::

