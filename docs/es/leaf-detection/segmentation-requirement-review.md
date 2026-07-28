# Revisión técnica del requerimiento de segmentación

Revisión integral previa al consumo de GPU en la nube, realizada el 2026-07-28.
No se entrenó ningún modelo, no se instalaron dependencias y no se descargaron
pesos. Cada cifra se verificó contra el repositorio, no contra la documentación
existente.

La evidencia tabular está en `outputs/leaf_detection/requirement_review/`.

## 1. Estado verificado

Todos los valores declarados en el requerimiento resultaron correctos:

| Dato | Declarado | Verificado | Método |
|---|---|---|---|
| Imágenes | 1 155 | 1 155 | conteo en `all/` y en los tres splits |
| Máscaras | 1 224 | 1 224 | líneas no vacías en todos los TXT |
| Clase | `0 = maize_leaf` | `{0: 1224}` | `validate_segmentation_dataset` |
| Fingerprint padre | `c087af60…9e38c` | idéntico | `make leaf-segmentation-verify-locks` |
| Fuentes | 155 / 1 000 | 155 / 1 000 | agregación de `split_manifest.csv` |
| Splits | 809/173/173 | 809/173/173 | `find` sobre `images/{split}` |
| Máscaras por split | 858/183/183 | 858/183/183 | conteo de líneas |
| Grupos | 1 035 | 1 035 | `group_id` distintos |
| Fingerprints de split | los cuatro | los cuatro | recálculo sobre 1 155 filas |
| `dataset_lock.status` | `ready_for_split_generation` | igual | lectura del lock |
| `split_lock.status` | `ready_for_training_preflight` | igual | lectura del lock |
| Paquete | 2 132 850 255 B | idéntico | `stat` |
| SHA-256 del paquete | `5d4d2bb6…b999` | idéntico | `sha256sum` recalculado |
| *(paquete reconstruido)* | — | `ec5dac44…dd6d` | ver V-03 |
| *(paquete v2 para upload/train)* | — | `4886ef3a…805c` | ver V-04 |
| Ultralytics local | no instalado | `ModuleNotFoundError` | import |
| `processing_profile` | `baseline_full` | igual | `config/dataset.yaml` |
| `leaf_detection.enabled` | `false` | igual | `config/dataset.yaml` |
| Entrenamiento | no ejecutado | sin `best.pt` ni `last.pt` | ausencia de `segmenter/` |

El piloto sigue intacto: 100 imágenes, sin participación en train, val, test ni
en ninguna selección. Los reportes de fuga contra el piloto están vacíos.

### Dos discrepancias menores y una relevante

**V-01 y V-02 (cosméticas).** `data/leaf_detection/pilot/annotations.xml` es un
symlink a `annotations/cvat/annotations.xml`. `Path.rglob` lo cuenta como
archivo (211) y `find -type f` no (210); por eso el manifiesto de transporte
declara 23 666 bytes de más respecto al lock. Los 211 archivos del manifiesto
existen en disco con hash y tamaño idénticos: no hay pérdida ni corrupción.

**V-03 (relevante, resuelto).** El paquete construido a las 09:19 contenía
versiones anteriores de `run_ultralytics.py`, `lib.sh` y `Makefile`. Se
reconstruyó con el código corregido: nuevo SHA-256
`ec5dac4478f43a83e9afeca3734131041022cac1611b76942de914e5ba93dd6d`
(2 132 860 866 B). El contenido se verificó por extracción contra el repositorio
y el determinismo se confirmó con dos construcciones independientes que
produjeron el mismo hash.

**V-04 (ruta crítica corregida).** Tras corregir propagación de `CONFIG`,
AutoBatch, gates de pérdidas/GPU/checkpoints, locks canónicos, bootstrap,
preflight e identidad reanudable del run, se generó el paquete
`v2-c087af60-seed42`: SHA-256
`4886ef3a11edb5d4819b9e980981a3f697f85129238a0b25e78eb9b0bc82805c`
(2 132 873 091 B). Dos reconstrucciones consecutivas produjeron el mismo hash;
la extracción verificó 2 402 checksums, 2 401 archivos contra el árbol fuente
y cero rutas prohibidas.

## 2. Historia técnica reconstruida

La cronología declarada es correcta en sus quince puntos. Se conservan los
errores y resultados negativos porque son parte de la evidencia:

1. **Sospecha de atención al fondo.** Motivó todo el requerimiento.
2. **Piloto manual con CVAT.** 100 imágenes reales del test oficial, semilla 42,
   una caja `maize_leaf` por imagen.
3. **Fallo del exportador.** El export YOLO produjo sólo 48 TXT y omitió 52
   cajas rotadas. Se recuperaron desde el XML nativo; 36 necesitaron clipping.
   El manifiesto final tiene 99 `annotated` y `image_0021` como `ambiguous`
   (área 0.092799, por debajo del mínimo 0.15). No se expandió artificialmente.
4. **Diagnóstico full frente a ROI.** Tres checkpoints históricos perdieron
   macro-F1: EfficientNet-B0 `0.8827→0.8561`, ShuffleNetV2 `0.9064→0.7582`,
   EfficientNet-Lite0 `0.9052→0.6101`.
5. **Interpretación conservadora.** La caída se atribuyó al cambio de
   distribución en inferencia (recorte, escala, letterbox, padding, pérdida de
   contexto), no a que aislar hojas sea inútil. **Ese resultado negativo sigue
   siendo el argumento principal para no aplicar YOLO sólo en inferencia.**
6. **`baseline_full` se mantuvo.** `leaf_detection.enabled=false`, sin cambios.
7. **Giro a segmentación de instancias.** Una caja conserva fondo rectangular;
   una máscara sigue el contorno.
8. **Dos fuentes externas.** 1 003 y 157 imágenes con respaldos COCO.
9. **EDA con correcciones.** Detectó 11 bbox mezclados en el export de
   segmentación, 8 autointersecciones y 1 vértice repetido; sustituyó una caché
   débil por fingerprints SHA-256 de 2 428 archivos.
10. **Error del renderer de previews.** Construía cada caso con `polygons=[]`,
    mostrando `instances=0` aunque el TXT tuviera geometría. Se regeneraron las
    35 previews desde las anotaciones originales.
11. **Decisiones humanas.** 35 casos únicos: 16 `approved`, 16 `exclude`, 3
    `needs_reannotation`, sin contradicciones.
12. **Consolidación.** De 1 160 candidatas quedaron 1 155; se excluyeron 13 392
    anotaciones de lesión; una máscara se recuperó desde COCO y otra, de área
    `4.28e-7`, se envió a reanotación.
13. **Splits por grupos**, **14. paquete cloud** y **15. sin entrenar**:
    verificados arriba.

Una cifra de la documentación previa merece nota: la historia menciona "1 156
imágenes y 1 226 máscaras" en la consolidación inicial y luego 1 155/1 224. La
diferencia corresponde a la exclusión posterior de la recuperación COCO
extremadamente pequeña. Ambas cifras son correctas en su momento y la
trazabilidad está en `reannotation_queue.csv`.

## 3. Auditoría de datos y splits

Verificado sin incidencias: correspondencia imagen-etiqueta 1:1, clase única,
sintaxis y topología válidas, cero duplicados exactos, cero variantes Roboflow
cruzadas, 1 035 grupos indivisibles, cero fugas entre splits y cero fugas contra
el piloto (umbral perceptual Hamming ≤ 4).

### La brecha de dominio es de adquisición, no de clases

Con una sola clase, la diferencia 155 frente a 1 000 no es desbalance de clases.
Es una brecha de dominio, y al medirla resulta más profunda de lo que sugiere el
conteo:

| Rasgo | `corn` (155) | `corn_leaf_diseases_classification` (1 000) |
|---|---|---|
| Resoluciones distintas | **1** (224×224) | **35** |
| Megapíxeles (mediana) | 0.05 | 9.14 |
| Orientación | 100 % cuadrada | 41 % cuadrada, 31 % vertical, 27 % horizontal |
| Área de máscara (mediana) | 0.316 | 0.463 |
| Imágenes multi-hoja | 31 de 155 (20.0 %) | 13 de 1 000 (1.3 %) |

Las 155 imágenes de `corn` ya vienen redimensionadas a 224×224, así que no
aportan detalle fino; aportan variedad de composición (una de cada cinco tiene
varias hojas, frente a una de cada setenta y siete en la fuente grande).

El dato que más importa es el tercero: **el piloto retenido —que representa el
dominio real de DoctorMaiz— tiene 39 resoluciones distintas entre 0.01 y 20.16
MP**. No se parece a ninguna de las dos fuentes de entrenamiento. Esto refuerza
por qué el piloto debe permanecer retenido y por qué su resultado no debe
anticiparse desde las métricas internas.

Riesgos derivados: dependencia de la fuente grande, bajo rendimiento en el
dominio pequeño, y sobre todo que el modelo aprenda a segmentar hojas grandes y
centradas —el patrón dominante— y falle en composiciones multi-hoja, que son
justamente donde `corn` aporta señal.

Propuestas, **ninguna aplicada** (modificar splits exige decisión formal y nuevo
fingerprint): métricas separadas por fuente en cada evaluación, muestreo
balanceado como ablación D-03, y un experimento leave-one-source-out sólo si
D-03 muestra que la fuente pequeña queda desatendida.

## 4. Auditoría de máscaras

La geometría es válida, pero validez no es adecuación. Los datos estructurales:

- **1 001 de 1 155 imágenes (86.7 %) tienen máscaras que tocan el borde del
  encuadre.** Es esperable en fotografía de hoja cercana, y significa que el
  modelo aprenderá muchas hojas truncadas por el marco.
- Sólo **18 imágenes** tienen alguna máscara pequeña (< 0.05 de área) y **435**
  tienen alguna grande (> 0.50). La distribución está sesgada a hojas grandes:
  el 62 % de las máscaras cae en el bin medio y el 35.5 % en el grande.
- **44 imágenes (3.8 %)** son multi-instancia.

Lo que **no** puede verificarse sin entrenar: si las máscaras cubren la hoja
completa, si pierden puntas, si incluyen tallo o fondo, o si incluyen parte de
otra hoja. Eso requiere comparar predicciones contra ground truth.

### La asimetría que define las prioridades de DoctorMaiz

**Una máscara que corta tejido enfermo es peor que una ligeramente amplia.** El
segmentador existe para alimentar a un clasificador de enfermedades y
deficiencias nutricionales:

- Si la máscara **recorta** parte de la hoja, puede eliminar precisamente la
  lesión o la zona clorótica que define el diagnóstico. La información se pierde
  y el clasificador no puede recuperarla. Las deficiencias de nitrógeno,
  fósforo y potasio se manifiestan como patrones a lo largo del limbo y en los
  márgenes: perder bordes es perder el síntoma.
- Si la máscara es **algo amplia**, entra algo de fondo. Eso es exactamente lo
  que el clasificador ya sabe manejar hoy (`baseline_full` trabaja con la imagen
  completa) y sólo reduce parcialmente el beneficio esperado.

El error de recorte destruye señal; el error de exceso sólo diluye la mejora.
Por eso el orden de prioridad de métricas es:

**Prioridad alta:** recall de píxel de hoja (qué fracción del tejido real
sobrevive), under-segmentation ratio, porcentaje de hoja recortada, imágenes sin
detección y tasa de fallback.

**Prioridad media:** IoU y Dice (equilibrio global), tasa de máscara principal
incorrecta, over-segmentation ratio, cobertura de bordes.

**Prioridad informativa:** mask precision, mAP50 y mAP50-95, tiempo por imagen.
mAP es la métrica que Ultralytics optimiza y reporta, pero **penaliza igual el
exceso y el defecto**, de modo que un modelo con buen mAP puede ser peor
downstream que otro con mAP algo menor y mejor recall de tejido. Ese es el
riesgo R-14 del registro.

## 5. Modelo candidato

`yolo26n-seg.pt` con `ultralytics==8.4.104` no es verificable localmente:
Ultralytics no está instalado y el preflight lo registra honestamente como
`not_locally_verifiable` en vez de inferir soporte a partir del nombre. Esa
honestidad es correcta y debe conservarse.

La confirmación (configuración, pesos, licencia, forward que produzca máscaras,
exportabilidad) corresponde a la Fase A remota. La matriz de candidatos para
pruebas futuras está en `experiment_matrix.csv`: nano preentrenado como baseline
(D-05 controla el aporte del preentrenamiento), small sólo si el baseline
muestra subajuste. **Con 809 imágenes de entrenamiento, un modelo mayor tiene
más riesgo de sobreajuste que de mejora**, así que la decisión inicial debe
seguir siendo el nano.

## 6. Defectos encontrados en el código cloud

Tres defectos que sólo se habrían manifestado tras consumir GPU (todos
corregidos, con prueba, y ya incluidos en el paquete reconstruido):

1. **`selected_batch` y `loss` se leían de `result.trainer`.** El objeto que
   devuelve `YOLO.train()` son las métricas, no el trainer, así que
   `getattr(result, "trainer", None)` daba `None`: la configuración final del
   baseline habría conservado `batch=-1` en vez del batch medido —el propósito
   central del smoke— y `loss_items`, al ser un tensor, habría roto la
   serialización JSON del resumen después de horas de entrenamiento.
2. **Los scripts posteriores al bootstrap no reutilizaban `.venv-cloud`.**
   `bootstrap_cloud.sh` crea el entorno y lo activa sólo dentro de su propio
   proceso. Preflight, smoke y train arrancaban con `PYTHON=python`, que en una
   sesión nueva no tiene Ultralytics.
3. **`make clean-outputs` ejecutaba `rm -rf outputs/` sin confirmación**, lo que
   borraría el paquete de 2.13 GB, toda la evidencia de auditoría y, más
   adelante, los checkpoints entrenados.

Los tres están corregidos con pruebas. El detalle está en
`improvement_backlog.csv` (IMP-01 a IMP-03).

## 7. Contradicciones documentales

Ocho correcciones aplicadas y dos casos documentados sin tocar. Las principales:
la fase 10 figuraba como pendiente cuando los splits ya existían; se citaba un
objetivo `make leaf-segmentation-train` que no existe; el avance afirmaba que el
gate seguía bloqueado cuando ya se había desbloqueado; y los próximos pasos
seguían apuntando a revisiones manuales completadas.

Un caso se dejó sin modificar deliberadamente: `detector_dataset/test/` (copia
del piloto histórico) y `detector_dataset/images/test/` (test interno del
segmentador) tienen nombres casi idénticos. Renombrar cambia rutas históricas,
así que requiere decisión formal (IMP-16).

También quedó documentada, sin corregir, la divergencia entre la configuración
que propuso el preflight local (`cache=disk`, proyecto
`segmentation_training`, nombre `yolo26n_seg_full_seed42`) y la configuración
oficial del paquete (`cache=false`, `segmenter`, `yolo26n_seg_baseline`). Los
archivos del preflight son evidencia generada y regenerarlos sobrescribiría el
registro histórico.

## 8. Conclusión

**Listo:** dataset, splits, locks, fingerprints, aislamiento del piloto,
protecciones del Makefile, determinismo del empaquetado, trazabilidad completa,
paquete reconstruido con el código corregido, gate de un solo uso del test
interno y métricas downstream implementadas con pruebas.

**Antes del smoke:** verificar en la Fase A que `ultralytics==8.4.104` soporta
`yolo26n-seg` y ejecuta un forward que produzca máscaras. Es el único bloqueador
que queda y requiere una GPU real.

**Después del smoke:** decidir `cache`, congelar el batch medido, ajustar
`workers` a la máquina real y estimar coste con datos reales.

**Requiere GPU real:** todo lo relativo a batch, VRAM, velocidad, utilización y
el equilibrio entre `deterministic=true` y rendimiento.

**Debe esperar al baseline:** ablaciones, métricas downstream, política de
inferencia y fondo neutral.

**No debe cambiarse todavía:** splits, decisiones humanas, piloto,
`baseline_full`, `leaf_detection.enabled=false` y los hiperparámetros oficiales.
