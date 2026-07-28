# ADR: estrategia de entrenamiento y evaluación del segmentador

- **Fecha:** 2026-07-28
- **Estado:** aceptado para las fases A–C; las fases D–E quedan propuestas
- **Contexto previo:**
  [ADR de segmentación de instancias](adr-leaf-instance-segmentation-strategy.md),
  [ADR de datasets externos](adr-external-leaf-segmentation-datasets.md),
  [ADR del diagnóstico ROI](adr-manual-roi-diagnostic-result.md)

## Contexto

El dataset del segmentador está congelado y verificado: 1 155 imágenes, 1 224
máscaras de clase única `0 = maize_leaf`, fingerprint
`c087af60c2bad1c133c4ea8b14cee945405bfe4976aa80c4faf089d7a4b9e38c`, splits
809/173/173 sobre 1 035 grupos indivisibles y cero fugas, incluidas cero fugas
contra el piloto retenido.

Falta decidir cómo se entrena, cómo se evalúa y qué evidencia se exige en cada
paso antes de consumir GPU. Tres hechos condicionan la decisión:

1. **El diagnóstico histórico dejó un resultado negativo que sigue vigente.**
   Aplicar ROI sólo en inferencia sobre clasificadores entrenados con imagen
   completa degradó el macro-F1 hasta `0.9052→0.6101`. Se interpretó como cambio
   de distribución, no como prueba de que aislar hojas sea inútil.
2. **El conjunto de entrenamiento es pequeño y heterogéneo.** 809 imágenes de
   train, con una fuente de 155 imágenes todas a 224×224 y otra de 1 000 con 35
   resoluciones y mediana de 9.14 MP.
3. **El objetivo no es maximizar mAP.** El segmentador alimenta a un clasificador
   de enfermedades y deficiencias nutricionales.

## Decisión

### 1. Baseline pequeño y un solo factor por experimento

`yolo26n-seg` preentrenado, `imgsz=640`, `epochs=150`, `patience=30`,
`optimizer=auto`, `seed=42`, `deterministic=true`, `cache=false`, `device=0`.
Con 809 imágenes de entrenamiento, un modelo mayor tiene más riesgo de
sobreajuste que de mejora. Las ablaciones cambian **un** factor cada una; de lo
contrario la mejora no es atribuible.

`batch=-1` se acepta **sólo como punto de partida del smoke**. El batch efectivo
de AutoBatch depende de la VRAM libre del momento, lo que impide comparar
corridas entre máquinas. El valor medido en el smoke se congela como entero en
la configuración del baseline.

### 2. La prioridad de las métricas se alinea con el clasificador, no con mAP

Se adopta explícitamente esta asimetría: **una máscara que corta tejido enfermo
es peor que una ligeramente amplia.** Recortar destruye la lesión o la zona
clorótica que define el diagnóstico y esa información no se recupera; exceder
sólo introduce algo de fondo, que es lo que `baseline_full` ya maneja hoy.

En consecuencia se priorizan recall de píxel de hoja, under-segmentation,
porcentaje de hoja recortada, imágenes sin detección y tasa de fallback por
encima de mAP50-95. mAP se reporta porque es lo que Ultralytics optimiza, pero
**no** decide por sí solo: penaliza igual el exceso y el defecto.

### 3. Tres niveles de evaluación con separación estricta

- **Nivel 1 — val:** durante todo el desarrollo, reejecutable sin límite. Es el
  único conjunto que puede elegir hiperparámetros.
- **Nivel 2 — test interno:** una sola vez, con la configuración ya congelada.
  Tras verlo no se ajusta nada.
- **Nivel 3 — piloto externo:** después del test interno, como prueba de
  generalización a un dominio distinto. **Mientras el piloto conserve la regla
  histórica de una hoja principal por imagen, sólo admite revisión cualitativa**:
  calcular mAP contra anotaciones de hoja principal penalizaría como falso
  positivo cada hoja correctamente detectada pero no anotada.

El piloto representa el dominio real de DoctorMaiz (39 resoluciones entre 0.01 y
20.16 MP) y no se parece a ninguna de las dos fuentes de entrenamiento. Ese es
justamente su valor y la razón de no anticipar su resultado.

### 4. Siete gates formales

| Gate | Nombre | Exige |
|---|---|---|
| 1 | Cloud ready | GPU, CUDA, dependencias fijadas sin sustituir torch, ambos locks, pesos con hash registrado, forward que produzca máscaras, almacenamiento persistente verificado |
| 2 | Smoke approved | una época completa, pérdidas finitas, GPU realmente utilizada, checkpoints escritos, reanudación posible, val ejecutable, memoria estable, batch registrado |
| 3 | Baseline completed | entrenamiento finalizado, `best.pt` y `last.pt`, métricas de val, configuración congelada, logs y hashes |
| 4 | Configuration frozen | ablaciones completadas, configuración elegida **sólo con val**, confirmación multi-semilla, sin haber tocado test ni piloto |
| 5 | Internal test completed | evaluación única, métricas, análisis de errores, sin reajuste posterior |
| 6 | External pilot completed | evaluación externa, generalización, fallbacks medidos, análisis visual |
| 7 | Downstream ready | máscaras congeladas y versionadas, manifiestos, bbox ROI, masked ROI, splits de clasificación intactos |

### 5. Comparación downstream con representación consistente

Cuando llegue la fase del clasificador se compararán `baseline_full`,
`baseline_bbox_roi` y `baseline_masked_roi`, **cada uno entrenado y evaluado con
su propia representación**, manteniendo constantes splits, semilla,
arquitectura, epochs, optimizador y métricas. Arquitectura inicial:
EfficientNet-B0, la más robusta al cambio de representación en el diagnóstico
histórico. No se repetirá el error de aplicar YOLO sólo en inferencia.

### 6. Sin cambios en producción

`processing_profile=baseline_full` y `leaf_detection.enabled=false` se mantienen
hasta que exista evidencia downstream. Ningún resultado del segmentador, por
bueno que sea en mAP, justifica activar la ruta ROI sin la comparación del
punto 5.

## Alternativas consideradas y descartadas

- **Empezar por `yolo26s-seg`.** Descartada: con 809 imágenes de train el riesgo
  de sobreajuste supera la mejora esperable. Queda como ablación D-04, sólo si
  el baseline muestra subajuste.
- **Rebalancear los splits por fuente antes de entrenar.** Descartada: los
  splits están congelados con fingerprint y cambiarlos exige una decisión formal
  nueva. El balance por fuente se explorará como ablación D-03 sobre el
  muestreo, no sobre los splits.
- **Activar `cache=disk` desde el inicio** (como sugería el preflight local).
  Descartada por ahora: su beneficio depende de si el cuello de botella está en
  el DataLoader, lo que sólo el smoke puede establecer.
- **Calcular mAP sobre el piloto.** Descartada mientras el piloto conserve la
  regla de hoja principal, por la razón del punto 3.
- **Fijar umbrales numéricos de aceptación ahora.** Descartada: sin una sola
  corrida cualquier número sería arbitrario. Se fija la forma de los criterios y
  su orden de prioridad; los valores se proponen tras el baseline.

## Consecuencias

**Positivas.** Cada decisión queda atribuible a un factor y a una evidencia. El
test interno y el piloto conservan su valor estadístico. El coste cloud se
conoce antes de comprometerlo, porque el smoke precede al baseline. La
comparación downstream responde la pregunta original en vez de repetir el
diagnóstico fallido.

**Negativas.** El proceso es más lento que entrenar y comparar directamente: un
factor por experimento y confirmación multi-semilla multiplican las corridas. Se
acepta ese coste porque el diagnóstico histórico ya mostró lo que ocurre cuando
se cambia la representación sin controlar las variables.

**Riesgos asumidos.** El soporte de `yolo26n-seg` en `ultralytics==8.4.104` no
está verificado y sólo la Fase A puede confirmarlo. La brecha de dominio entre
fuentes puede limitar la generalización al piloto. Ambos están registrados en
`risk_register.csv` (R-06 y R-01).

## Evidencia

- `outputs/leaf_detection/requirement_review/` (matriz de trazabilidad, riesgos,
  cuellos de botella, backlog, augmentations, matriz experimental, estimador de
  coste, inconsistencias documentales y validación)
- `docs/es/leaf-detection/segmentation-requirement-review.md`
- `docs/es/leaf-detection/segmentation-training-optimization-plan.md`
- `docs/es/leaf-detection/segmentation-current-flow.md`
- `cloud_training/CLOUD_READINESS_CHECKLIST.md`
