# Modelo base y protocolo experimental

La arquitectura elegida es `yolo26n-seg`, variante nano de segmentación de instancias de la familia YOLO26 [@jocher2026yolo26]. El checkpoint inicial preentrenado sirve como punto de partida; la tarea del proyecto se redefine a una clase `maize_leaf`. La configuración base usa `imgsz=640`, máximo 150 épocas, paciencia 30, optimizador automático, semilla 42, ejecución determinista, ocho workers y caché desactivada en cloud.

El baseline C-01 se entrenó con batch 26. El registro de planificación identifica la mejor época como 87 y el pico de Mask mAP50-95 de validación como 0.93806. Su checkpoint es el archivo operacional `outputs/leaf_detection/models/doctor_maiz_leaf_segmenter_best.pt`, de 6.55 MB y hash `4f66456d…f9c5ec6f`. Este es el modelo que posteriormente procesó el corpus de 33,437 imágenes.

El orden de selección de experimentos prioriza: recall medio de píxel foliar en imágenes de una hoja, Dice medio en ese mismo subconjunto, Mask mAP50-95 y precisión externa del quality gate. Esta jerarquía reconoce que la métrica nativa del detector no mide por sí sola la pérdida de tejido diagnóstico después del recorte.

## Ablaciones previstas

Table: Plan de mejora definido antes de consumir test.

| ID | Cambio principal | Estado registrado |
|---|---|---|
| D-01 | Desactivar `mosaic` | Completado |
| D-02 | Reducir entrada a 512 | Preparado |
| D-02B | Aumentar entrada a 768 | Requiere smoke específico |
| D-03 | Balance por fuente | Preparado |
| D-04 | Cambiar a `yolo26s-seg` | Requiere pesos verificados y smoke |
| D-05 | Entrenar desde cero | Preparado |
| D-06 | Activar copy-paste | Preparado |
| E-01 | Repetir ganador con semillas 7 y 1337 | Pendiente |

Cada perfil cambia una hipótesis concreta y conserva los splits. D-04 no descarga pesos implícitamente; exige registrar un `yolo26s-seg.pt` suministrado y verificable. D-02B requiere medir primero el batch porque la resolución incrementa memoria. La comparación de D-01 con C-01 comparte semilla 42, lo que aísla mejor el efecto de mosaic, pero todavía no estima variación entre semillas.

El protocolo exige escoger el ganador únicamente con validación, repetirlo con semillas 7, 42 y 1337, y reportar media y desviación estándar. La mejora debe superar la variación observada. Sólo entonces puede congelarse configuración y abrirse test una vez. Esta secuencia evita que una ganancia pequeña de una sola corrida se interprete como robusta.

::: evidence Identidad de modelos
C-01 y D-01 no son intercambiables. C-01 tiene hash `4f66456d…`; D-01 tiene hash `a2bf4f20…`. Toda métrica o inferencia del informe conserva el hash del checkpoint que realmente la produjo.
:::

