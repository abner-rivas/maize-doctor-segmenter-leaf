# Plan de optimización del entrenamiento del segmentador

Complementa la [revisión del requerimiento](segmentation-requirement-review.md).
Ninguna optimización de este documento se aplicó automáticamente: las que
afectan resultados de entrenamiento requieren aprobación explícita y, varias,
evidencia de una GPU real.

## 1. Perfiles de ejecución

`deterministic=true` desactiva la selección de algoritmos cuDNN más rápidos.
Esa penalización es el precio de poder comparar dos configuraciones y atribuir
la diferencia al factor cambiado y no al ruido. Por eso se proponen dos perfiles
y una regla para elegirlos.

### Perfil reproducible (obligatorio para el baseline y toda comparación)

`seed=42`, `deterministic=true`, `cache` según decisión posterior al smoke,
batch entero fijo medido en el smoke. Es el perfil de `train_yolo26n_seg.yaml`.

### Perfil rápido (sólo ingeniería)

`deterministic=false`, batch mayor, `workers` al máximo de la máquina. Sirve
para comprobar que una tubería funciona, medir velocidad o depurar. **Sus
resultados no entran en ninguna comparación formal ni en el registro de
experimentos como métricas válidas.**

Regla: si el número va a aparecer en una tabla comparativa, se produjo con el
perfil reproducible. Si sólo responde "¿esto corre?", el perfil rápido sirve.

## 2. Mediciones que debe producir el smoke

El smoke actual registra duración, VRAM pico, métricas, pérdida, checkpoint y
batch seleccionado. Faltan las mediciones que permiten decidir todo lo demás:

| Medición | Para qué decide |
|---|---|
| imágenes/segundo | estimar el tiempo del baseline |
| tiempo por batch | detectar variabilidad por decodificación |
| tiempo por epoch (train y val por separado) | alimentar el estimador de coste |
| utilización de GPU (media y mínima) | **si baja del 80 %, el cuello es el DataLoader** |
| VRAM máxima | margen para congelar el batch |
| CPU y RAM | si `workers` está mal dimensionado |
| espera del DataLoader | confirma o descarta el cuello de datos |
| lectura de disco | distingue disco lento de decodificación lenta |

La utilización de GPU es la medición que dispara las demás decisiones: si es
alta, el modelo está limitado por cómputo y `cache` no aporta; si es baja, el
cuello está en leer y decodificar.

## 3. DataLoader y `cache`

`cache=false` obliga a decodificar en cada epoch imágenes cuya mediana es de
9.14 MP en la fuente grande. Sobre 150 epochs eso es mucha decodificación
repetida.

- `cache=disk` guarda los tensores redimensionados. Con 809 imágenes de train a
  640 px el coste en disco es modesto y elimina la decodificación repetida.
- `cache=ram` no es prudente sin conocer la RAM de la máquina remota.
- `cache=false` es la opción actual y la más conservadora.

**No se cambia ahora.** Es un parámetro de la configuración oficial y su efecto
no es medible sin GPU. La decisión se toma tras el smoke y sólo si la
utilización de GPU está por debajo del 80 %.

Lo mismo aplica a `workers=8`, fijado a ciegas: debe derivarse de los vCPU de la
máquina remota. Y a `pin_memory`, que Ultralytics gestiona internamente y no
conviene forzar.

## 4. Batch y resolución

### Batch

`batch=-1` (AutoBatch) es un punto de partida razonable, no una decisión final.
Su problema es que **el batch efectivo depende de la VRAM libre en ese momento**,
de modo que dos corridas en máquinas distintas —o en la misma con otra carga—
no son comparables.

Prueba propuesta para la nube, sin ejecutar todavía:

1. Dejar que AutoBatch resuelva un valor en el smoke y registrarlo.
2. Probar ese valor y el inmediatamente superior en potencia de dos, midiendo
   VRAM pico y tiempo por batch durante unas pocas iteraciones.
3. Elegir el mayor batch que deje al menos un 15 % de VRAM libre, contando el
   pico de la validación, que Ultralytics ejecuta al final de cada epoch.
4. **Congelar ese entero en la configuración del baseline.**

### Resolución

`imgsz=640` es el valor actual. La evidencia disponible para justificarlo o
cambiarlo:

- Las máscaras son grandes: mediana de área relativa 0.44, sólo 18 imágenes con
  alguna máscara menor a 0.05. **No hay un problema de objetos pequeños que
  justifique subir de 640.**
- Las fuentes son muy dispares (0.05 MP frente a 9.14 MP de mediana): cualquier
  `imgsz` implica escalar mucho una de las dos. Las 155 imágenes de `corn`
  (224×224) se ampliarían casi 3× a 640 y algo menos de 2.3× a 512.
- La latencia futura importa: el segmentador precede al clasificador.

Por eso la ablación D-02 propone 512 como único factor cambiado. Si mantiene el
mAP de máscara, 512 reduce coste y latencia. Subir por encima de 640 no está
justificado con la distribución de tamaños actual.

## 5. Nomenclatura de corridas y política de reanudación

La configuración oficial usa `name: yolo26n_seg_baseline` fijo. Ultralytics no
sobrescribe —crearía `yolo26n_seg_baseline2`—, pero `validate.sh`,
`evaluate_test.sh` y `leaf_segmentation_make.py` apuntan a la ruta del primero,
así que una segunda corrida quedaría entrenada pero nunca evaluada (riesgo
R-20).

Nomenclatura propuesta para la fase de ablaciones:

```
<modelo>_<dataset_version>_<seed>_<timestamp>
yolo26n-seg_c087af60-splits-874b217b_seed42_20260728T1430Z
```

La política de reanudación debe distinguir cuatro operaciones que hoy no están
todas separadas:

| Operación | Punto de partida | Estado actual |
|---|---|---|
| Reanudar interrumpida | `last.pt` con `resume=True` | implementado |
| Nueva corrida desde `best.pt` | `best.pt` como pesos iniciales | **no distinguido** |
| Fine-tuning | `best.pt` con otra configuración | **no distinguido** |
| Evaluación | `best.pt` sin entrenar | implementado |

Además `resume_manifest.json` se sobrescribe en cada reanudación, de modo que
una corrida interrumpida tres veces conserva sólo el último registro (IMP-10).

## 6. Política de inferencia (diseño, sin integrar)

Reglas evaluables, no valores definitivos. Todos los umbrales deben fijarse con
val y nunca con test ni con el piloto.

- **Confidence threshold:** barrer en val y elegir el que maximice el recall de
  píxel de hoja sujeto a una tasa aceptable de máscaras falsas. Empezar bajo:
  perder una hoja es peor que detectar una de más.
- **IoU threshold (NMS):** relevante sólo en multi-hoja (3.8 % de imágenes).
  Un valor bajo fusionaría hojas adyacentes; conviene ser conservador.
- **Selección de máscara principal:** candidatas evaluables son mayor área,
  mayor confianza, y mayor área ponderada por centralidad. Debe medirse la tasa
  de máscara principal incorrecta sobre una muestra revisada de val.
- **Múltiples hojas:** decidir explícitamente entre analizar sólo la principal o
  unir todas las máscaras de la imagen. La segunda opción es más coherente con
  la regla de anotación multihoja del detector.
- **Dilatación y margen:** dado que recortar es peor que exceder, una dilatación
  pequeña (por ejemplo 2–5 px, a barrer en val) protege bordes y puntas.
- **Suavizado, huecos internos y componentes pequeños:** rellenar huecos
  internos es seguro (son parte de la hoja); eliminar componentes pequeños es
  arriesgado si corresponden a una punta separada por oclusión. Medir antes de
  activar.
- **Máscara mínima:** por debajo de un área relativa mínima, tratar la detección
  como no confiable y aplicar fallback.
- **Fallback:** cuando no exista máscara confiable, **usar la imagen completa,
  registrar el fallback en el manifiesto y nunca producir una predicción
  silenciosamente recortada de forma incorrecta.** La tasa de fallback es una
  métrica de primer nivel: un modelo con buen mAP y 20 % de fallback es peor en
  producción que uno algo inferior con 2 %.

## 7. Fondo neutral (ablación futura)

Variantes a comparar, todas con clasificador entrenado y evaluado con la misma
representación: negro, blanco, media del dataset, media de ImageNet, desenfoque
del fondo, alpha compositing, recorte bbox sin máscara y máscara con margen.

El objetivo declarado es evitar que el padding se convierta en un nuevo atajo
visual. El riesgo es concreto: si todas las imágenes de una clase acaban con más
proporción de fondo neutral que las de otra, el clasificador puede aprender la
proporción de relleno en lugar del síntoma. Por eso la ablación debe incluir al
menos una variante sin color constante (desenfoque) como control.

## 8. Relación con el clasificador

La comparación correcta, que el diagnóstico histórico ya demostró necesaria:

- `baseline_full`: entrenado y evaluado con imágenes completas (referencia
  actual, sin cambios).
- `baseline_bbox_roi`: entrenado y evaluado con bounding boxes derivados de las
  máscaras.
- `baseline_masked_roi`: entrenado y evaluado con máscaras y fondo neutral.

Constantes obligatorias: splits de clasificación (`data/splits/seed_42_baseline/`
intactos), semilla, arquitectura, epochs, optimizador y métricas. Arquitectura
inicial EfficientNet-B0: fue la más robusta al cambio de representación en el
diagnóstico histórico (−0.0266 de macro-F1 frente a −0.1482 y −0.2951).

**No repetir el error histórico:** aplicar YOLO sólo en inferencia sobre
checkpoints entrenados con imagen completa produce cambio de distribución y no
responde la pregunta. Cada variante se entrena con la representación que
recibirá.

Las máscaras deben congelarse antes de entrenar los clasificadores: generarlas
una vez, versionarlas con fingerprint y manifiesto, y no regenerarlas entre
variantes. Si el segmentador cambia, las tres variantes se rehacen juntas.

## 9. Paquete cloud

Composición medida: 2 316 archivos de dataset (2.335 GB) y sólo 2.78 MB de
código, scripts y documentación. El archivo comprimido pesa 2 132 850 255 B, un
8.8 % menos que el payload.

Opciones consideradas, ninguna aplicada:

- **Separar código y datos.** Es la de mayor impacto: el dataset es estable y se
  sube una vez; el código son 3 MB y podría corregirse sin resubir 2.1 GB. Esta
  revisión es la prueba del problema —tres correcciones de código invalidaron un
  paquete de 2.13 GB.
- **`tar.zst` o tar sin comprimir.** gzip aporta poco sobre JPEG y cuesta CPU en
  ambos extremos. Debe decidirse por tiempo total (comprimir + subir + extraer),
  no por tamaño, y sin sacrificar compatibilidad: `tar.gz` se abre en cualquier
  máquina, `zstd` no siempre está instalado.
- **Excluir los tres documentos** (11.8 KB) no aporta nada medible.

Lo que **no** debe recortarse: train, val, test, locks, manifiestos, scripts,
configuración ni trazabilidad. El paquete pesa lo que pesan las imágenes.

## 10. Estrategia de semillas

- `seed=42` para todo el desarrollo y todas las ablaciones.
- Tres semillas (42, 7, 1337) sólo para las configuraciones finalistas.
- No repetir todas las ablaciones con múltiples semillas: multiplicaría el coste
  sin cambiar el orden de las decisiones.

La regla de interpretación: **si la diferencia entre dos configuraciones es
menor que la desviación entre semillas de una misma configuración, esa
diferencia no es evidencia.** Por eso E-01 precede a congelar la configuración.

Cada corrida se registra en `outputs/leaf_detection/segmenter/experiment_registry.csv`
con seed, commit, fingerprints de dataset y split, hash de pesos iniciales,
entorno, GPU, batch real, duración, coste estimado, checkpoints y métricas. El
registro es un CSV local: no obliga a instalar MLflow ni Weights & Biases,
aunque cualquiera de los dos podría adoptarse después sin perder el histórico.

## 11. Criterios de aceptación provisionales

No se fijan umbrales numéricos porque no hay evidencia para justificarlos: sin
una sola corrida, cualquier cifra sería arbitraria. Lo que sí puede fijarse
ahora es la **forma** de los criterios y el orden de prioridad:

1. El baseline debe superar de forma clara a un control trivial (por ejemplo,
   tomar la imagen completa como máscara) en IoU. Si no lo hace, el problema es
   de datos o de configuración, no de ajuste fino.
2. El recall de píxel de hoja debe ser alto y la tasa de fallback baja; ambos
   umbrales se fijan tras ver la distribución en val, no antes.
3. Las métricas por fuente no deben divergir tanto como para que el dominio
   pequeño quede inservible.
4. La latencia debe caber en el presupuesto de la cascada segmentador +
   clasificador, que todavía no está definido.

Los números concretos se proponen tras el baseline y se congelan antes de tocar
el test interno.
