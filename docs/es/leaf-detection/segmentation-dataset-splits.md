# Splits del dataset de segmentación de hojas

## Padre congelado

La única entrada es `data/leaf_detection/detector_dataset/all/`. Antes de
dividir se exigió `dataset_lock.status=ready_for_split_generation` y se
recalculó el fingerprint:

`7a4a5c083fc64b067df12bcc95ec976d5a7e3b8a585d0a090b6b3940af4d7d5c`

El padre contiene 1 155 imágenes, 1 155 TXT y 1 224 máscaras de clase única
`0 = maize_leaf`. `all/`, `external_sources/` y el piloto son entradas
protegidas y no se materializan por movimiento.

El fingerprint padre anterior,
`c087af60c2bad1c133c4ea8b14cee945405bfe4976aa80c4faf089d7a4b9e38c`,
cambió porque `cldc_c6f46aaea98a271c.jpg` procedía de un JPEG decodificable
pero sin marcador EOI. La fuente original conserva SHA-256
`c6f46aaea98a271c11478592bb16ba101201345d3f4d9ed2dde158f5e1551561`;
la copia derivada canónica aplica `append_ffd9` sin recodificar y pasa a
`0d4d3554fa1b0fbf2d02c2d083cb93d3f601b643419942a19bfaa579e7b30888`.
El hash de píxeles decodificados antes y después es
`d63697cd3d48aef376302d841baaae2a666fd200d39c05955a8173ee740f70c6`,
con modo normalizado RGB y dimensiones 3048×4064. Se incorporó la versión
extendida de `image_normalization_manifest.csv` al fingerprint. Las decisiones
humanas, las 1 155 identidades lógicas y su asignación de split no cambiaron.

## Método

La semilla es 42 y las proporciones objetivo son 70/15/15. Primero se
construyen componentes conexos deterministas con estas señales:

- fuente y nombre original anterior al sufijo Roboflow;
- `roboflow_variant_group` y `duplicate_group`;
- SHA-256 exacto;
- hash perceptual promedio de 64 bits, usando distancia Hamming menor o igual
  a 4.

Las 1 155 imágenes forman 1 035 grupos indivisibles; el mayor contiene 12
imágenes. Después se minimiza la desviación de imágenes, máscaras, fuente,
instancias, bins de área, orientación, resolución, contacto con bordes e
imágenes multiinstancia. Un refinamiento por intercambio de grupos del mismo
tamaño mejora el balance sin cambiar los conteos por split.

## Resultado

| Split | Imágenes | Objetivo | Real | Diferencia | Máscaras | Grupos |
|---|---:|---:|---:|---:|---:|---:|
| train | 809 | 70.0000 % | 70.0433 % | +0.0433 pp | 858 | 724 |
| val | 173 | 15.0000 % | 14.9784 % | -0.0216 pp | 183 | 156 |
| test | 173 | 15.0000 % | 14.9784 % | -0.0216 pp | 183 | 155 |

| Split | `corn` | `corn_leaf_diseases_classification` |
|---|---:|---:|
| train | 109 | 700 |
| val | 23 | 150 |
| test | 23 | 150 |

Los fingerprints SHA-256 de membresía y contenido son:

- train:
  `06035eed94b920b9c7ad600d76eec132b93ade78ace1edb7d6a48340085d29ba`;
- val:
  `3c7bf7aba8a9f29b409c61bad4d9e9d59a3387915592f181ad3950ac8374e720`;
- test:
  `046545351ce79431bb1a995dfbc7dfa44c642a18a046860ed5edb9fc0ed89c51`;
- asignación combinada:
  `96833e43a46c959f0d5c86615b1d1ea6deecb139063eea9c877986a61084c0e1`.

El fingerprint combinado codifica la membresía de split; por ello no pretende
ser igual al fingerprint del árbol padre. `parent_content_equivalent=true`
confirma que la unión de los splits conserva las 1 155 imágenes y 1 224
máscaras verificadas del padre.

## Validación y reproducibilidad

El resultado tiene cero:

- hashes exactos compartidos;
- grupos compartidos;
- variantes Roboflow compartidas;
- pares perceptualmente cercanos entre splits dentro del umbral 4;
- cruces exactos, nominales, de base o perceptuales con el piloto;
- errores de sintaxis, clase, coordenadas, topología o correspondencia
  imagen/TXT.

Se ejecutó la generación dos veces en directorios temporales. Las asignaciones,
los manifiestos y los fingerprints fueron idénticos. La materialización final
usa copia; no hay symlinks ni hardlinks. `dataset.yaml` contiene rutas
relativas:

La reconstrucción tomó el mapa anterior `filename → split` como contrato y
comprobó los 1 155 nombres: hubo cero cambios de asignación. La validación
obligatoria cubrió los 2 310 JPEG canónicos de `all/images` y los tres splits:
cero SOI/EOI ausentes, cero errores de `Image.verify()` y cero errores de
`image.load()`. El checker real `ultralytics.data.utils.check_image` de
Ultralytics 8.4.104 recorrió esas 2 310 rutas en una copia temporal y modificó
cero hashes. Los 38 JPEG auxiliares sin EOI permanecen intactos y documentados
por separado en `auxiliary_jpeg_audit.csv`; no bloquean entrenamiento.

```yaml
train: images/train
val: images/val
test: images/test

names:
  0: maize_leaf
```

## Función de las evaluaciones

`test` es el test interno de las dos fuentes externas ya consolidadas. El
contenido de `data/leaf_detection/pilot/` no participa en train, val ni test y
permanece como evaluación externa retenida.

Los manifiestos y el gate están en
`data/leaf_detection/detector_dataset/manifests/`. Las estadísticas, reportes
de fugas, nueve gráficos y previews con máscaras superpuestas están en
`outputs/leaf_detection/detector_dataset_splits/`.

`split_lock.status=ready_for_training_preflight`. Este estado sólo habilita el
preflight posterior: durante esta fase no se entrenó ningún modelo, no se
instaló Ultralytics y no se descargaron pesos.

## Resultado del preflight posterior

El preflight volvió a calcular el fingerprint padre y los tres fingerprints de
split sin regenerarlos. También revalidó las 809/173/173 imágenes y
858/183/183 máscaras. El smoke loader rasterizó un batch 4/2/2 a 640 píxeles,
con tensores finitos de formas `[8, 3, 640, 640]` y `[8, 1, 640, 640]`.

La máquina local no tiene CUDA utilizable y Ultralytics no está instalado; por
ello el estado es `blocked_by_missing_dependency`. Consulte
[Preflight de entrenamiento](segmentation-training-preflight.md).
