# ADR: ROI manual no activado en modelos históricos

- Estado: aceptada
- Fecha: 2026-07-27
- Alcance: checkpoints históricos y procesamiento ROI

## Contexto

El clasificador podía aprender atajos relacionados con suelo, cielo, manos,
tallos, sombras, otras hojas y fondos de laboratorio. Como prueba de concepto
se implementó:

```text
imagen completa
→ bounding box manual
→ validación y clipping
→ margen
→ recorte
→ letterbox
→ normalización
→ clasificador
```

El piloto retenido contiene 100 imágenes del test oficial, 100 anotaciones CVAT
y 100 filas estructuralmente válidas. El XML nativo conserva 48 cajas directas
y 52 rotadas; 36 requirieron clipping. El manifiesto final registra 99 casos
`annotated` y `image_0021` como `ambiguous`, con área ROI `0.092799`.

## Diagnóstico

No se entrenaron modelos. Se aplicaron dos representaciones a los mismos
checkpoints:

- `baseline_full`: imagen completa y transformación histórica;
- `baseline_roi`: ROI manual, margen y letterbox.

| Modelo | Accuracy full | Accuracy ROI | Macro-F1 full | Macro-F1 ROI |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 0.8889 | 0.8586 | 0.8827 | 0.8561 |
| ShuffleNetV2-x1.0 | 0.9091 | 0.7576 | 0.9064 | 0.7582 |
| EfficientNet-Lite0 | 0.9091 | 0.6162 | 0.9052 | 0.6101 |

| Modelo | Predicciones cambiadas | Errores corregidos | Aciertos perdidos |
|---|---:|---:|---:|
| EfficientNet-B0 | 7 | 2 | 5 |
| ShuffleNetV2-x1.0 | 21 | 3 | 18 |
| EfficientNet-Lite0 | 39 | 2 | 31 |

## Interpretación

La caída no demuestra que “ROI no funciona”. Los clasificadores aprendieron con
imágenes completas y recibieron durante el diagnóstico otra escala, menos
contexto, recorte, letterbox y padding. Esa diferencia introdujo un cambio de
distribución entre entrenamiento e inferencia.

El clasificador es el componente que debe aprender a trabajar con la nueva
representación. El diagnóstico sólo demuestra que no es seguro cambiar la
entrada después del entrenamiento.

## Decisión

Se mantiene:

```yaml
processing_profile: baseline_full

leaf_detection:
  enabled: false
```

`baseline_roi` no se conecta a `CornDataset`, `predict.py`, LIME ni Grad-CAM.
Una comparación justa deberá entrenar y evaluar cada clasificador con la misma
representación.

## Consecuencias

- Los tres checkpoints y sus resúmenes permanecen inmutables.
- El piloto continúa retenido como test.
- El resultado no descarta el aislamiento de hojas.
- Se investigarán bounding boxes derivados de máscara y hojas segmentadas.

## Evidencia

- `data/leaf_detection/pilot/annotations/cvat/annotations.xml`;
- `data/leaf_detection/pilot/manifests/roi_manifest.csv`;
- `outputs/leaf_detection/pilot/diagnostic_experiment/`;
- `docs/es/preprocessed/manual-roi-diagnostic.md`.
