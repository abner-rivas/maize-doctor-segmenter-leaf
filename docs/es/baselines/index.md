# Baselines

Los baselines cumplen un doble propósito en este proyecto:

1. **Punto de comparación mínimo.** Cualquier arquitectura más compleja propuesta en etapas posteriores debe superar consistentemente estas cifras para justificar su mayor costo computacional o de mantenimiento.
2. **Demo de modelos candidatos.** Antes de comprometer el entrenamiento completo, los baselines permiten observar el comportamiento inicial de cada arquitectura candidata sobre una fracción representativa del dataset, detectar problemas (colapso de clases, overfitting temprano, incompatibilidad con el pipeline) a bajo costo.

Los tres modelos elegidos —**EfficientNet-B0**, **ShuffleNetV2-x1.0** y **EfficientNet-Lite0**— son redes convolucionales ligeras pre-entrenadas en ImageNet que cubren el eje precisión↔eficiencia y convierten a TFLite para despliegue móvil offline. Se priorizó este perfil porque el objetivo final es un sistema desplegable en entornos con recursos limitados (dispositivos móviles o cómputo en campo), y porque al tener parámetros comparables entre sí hacen que las diferencias de rendimiento sean atribuibles a la arquitectura, no al tamaño.

---

## Dataset utilizado

Los baselines se entrenan sobre el **perfil `baseline`** de `config/dataset.yaml`: las **9 clases**
del dataset con un tope de **1 500 imágenes por clase** ("cap la cabeza, conserva la cola" — solo
se recortan las clases mayoritarias, las minoritarias quedan intactas). Se genera con
`make splits-baseline` en `outputs/splits/seed_42_baseline/`, con la misma estratificación por
`label + environment` y seed 42 que el split completo.

| Split | Imágenes (aprox.) |
|---|---:|
| Entrenamiento (`train.csv`, 70 %) | ~7 014 |
| Validación (`val.csv`, 15 %) | ~1 503 |
| Prueba (`test.csv`, 15 %) | 1 503 |
| **Total** | **~10 020** |

El cap conserva completas las clases minoritarias (potasio 266, nitrógeno 523, fósforo 612) y
limita solo las mayoritarias (healthy, tizones, gusano cogollero), preservando el desbalance
natural sin gastar cómputo en imágenes redundantes de la cabeza. El cap es configurable desde
`config/dataset.yaml` (`baseline.max_images_per_class`) o por CLI (`--max-per-class`,
`--regenerate-splits` para forzar la regeneración). El modelo finalista se re-entrena sobre las
9 clases sin cap en el pipeline principal (`train.py`).

---

## Modelos seleccionados

### ShuffleNetV2-x1.0

**ShuffleNetV2-x1.0** es una CNN diseñada por Megvii (Face++) en 2018 explícitamente para
inferencia eficiente en dispositivos móviles <sup>[[17]](#ref-17)</sup>. Su contribución es un
conjunto de guías prácticas de diseño (no solo minimizar FLOPs, sino también el costo real de
memoria y acceso), materializadas en dos operaciones:

- **Channel split + channel shuffle:** divide los canales en dos ramas y, tras procesarlas,
  los baraja para que la información fluya entre grupos sin convoluciones densas costosas.
- **Sin convoluciones agrupadas 1×1:** evita el cuello de botella de acceso a memoria (MAC) que
  penalizaba a ShuffleNetV1, priorizando velocidad real sobre FLOPs teóricos.

En ImageNet-1K alcanza ~69 % de Top-1 con solo ~2.3 M de parámetros, siendo uno de los modelos
más pequeños del grupo (~5 MB serializado).

**Trade-offs relevantes para este proyecto:**

| Aspecto | Detalle |
|---|---|
| Precisión | Inferior a EfficientNet-B0 en ImageNet, pero competitiva tras fine-tuning en el dataset de maíz (macro-F1 0.9030 en 9 clases) |
| Velocidad / tamaño | El más ligero del grupo (~5 MB, ~2.3 M params); ideal para inferencia en gama baja |
| Transfer learning | Pre-entrenado con `ShuffleNet_V2_X1_0_Weights.DEFAULT`; se reemplaza la capa `fc` final |
| Despliegue | Operaciones (channel shuffle, depthwise) soportadas por TFLite; convierte y cuantiza sin ops exóticas |
| Riesgo | Capacidad limitada en clases visualmente ambiguas (deficiencias N/P/K), donde todos los baselines sufren |

Se construye con `torchvision` reemplazando `model.fc` por una `nn.Linear(in_features, 9)`.

---

### EfficientNet-B0

**EfficientNet-B0** es la red base de la familia EfficientNet, propuesta por Google Brain en 2019. Su contribución central es el *compound scaling*: en lugar de escalar solo la profundidad, el ancho o la resolución de entrada de forma independiente (como hacía la práctica anterior), EfficientNet escala los tres simultáneamente con un coeficiente compuesto $\phi$ determinado por búsqueda de arquitectura (NAS) <sup>[[8]](#ref-8)</sup>.

La versión B0 es el punto de partida de la familia: la arquitectura base encontrada por NAS antes de aplicar cualquier escala adicional. Usa bloques **MBConv** (Mobile Inverted Bottleneck) <sup>[[6]](#ref-6)</sup> con:
- Conexiones residuales
- Expansión de canales seguida de proyección
- Squeeze-and-Excitation integrado en cada bloque <sup>[[9]](#ref-9)</sup>

En ImageNet-1K alcanza ~77.1 % de Top-1 con 5.3 M de parámetros -más del doble que ShuffleNetV2-x1.0, pero con mayor precisión.

**Trade-offs relevantes para este proyecto:**

| Aspecto | Detalle |
|---|---|
| Precisión | Superior a ShuffleNetV2-x1.0 en ImageNet (~77 % vs. ~69 %); también lidera tras fine-tuning (macro-F1 0.9146 vs. 0.9030) |
| Velocidad de inferencia | Más lento que ShuffleNetV2-x1.0 en hardware móvil por las operaciones SE en cada bloque y el mayor número de parámetros |
| Tamaño del modelo | ~16 MB serializado; más pesado que ShuffleNetV2-x1.0 (~5 MB) |
| Transfer learning | Pre-entrenado en ImageNet con `EfficientNet_B0_Weights.DEFAULT` (IMAGENET1K_V1); se reemplaza `model.classifier[1]` |
| Regularización implícita | Los bloques MBConv con dropout estructural hacen a EfficientNet-B0 más robusto al overfitting con datasets pequeños |

---

### EfficientNet-Lite0

**EfficientNet-Lite0** es una variante de EfficientNet-B0 optimizada específicamente para dispositivos de borde con aceleradores de inferencia (Coral Edge TPU, microcontroladores ARM con CMSIS-NN) <sup>[[13]](#ref-13)</sup>. Las diferencias con B0 son:

- **Sin Squeeze-and-Excitation:** los bloques SE se eliminan porque su operación de reducción global no se mapea eficientemente en aceleradores de inferencia cuantizados.
- **ReLU6 en lugar de Swish:** más compatible con cuantización INT8, donde Swish introduce errores de representación no triviales que hacen caer la precisión de ~75 % a ~46 % si no se sustituye <sup>[[13]](#ref-13)</sup>.
- **Sin stem strided convolution:** la red evita algunas operaciones que rompen la compatibilidad con ciertos compiladores de modelos (TFLite, ONNX para Edge TPU).

Se construye con `timm` (`timm.create_model("efficientnet_lite0", pretrained=True, num_classes=9)`) porque `torchvision` no incluye esta variante.

**Trade-offs relevantes para este proyecto:**

| Aspecto | Detalle |
|---|---|
| Precisión | ~1–2 pp inferior a EfficientNet-B0 en ImageNet (~74–75 % Top-1); compensado por facilidad de despliegue |
| Velocidad de inferencia | Similar o superior a ShuffleNetV2-x1.0 en aceleradores compatibles; sin ventaja clara en GPU estándar |
| Cuantización | Diseñada para cuantizarse a INT8 sin degradación significativa; punto fuerte para despliegue en campo |
| Dependencia adicional | Requiere `timm` (no incluida en `torchvision`); añade una dependencia al entorno |
| Uso en proyecto | Representa el extremo del trade-off "máxima eficiencia en edge" para comparar contra el extremo "máxima precisión" de EfficientNet-B0 |

---

## Comparación de los tres modelos

| Modelo | Parámetros | Top-1 ImageNet | Tamaño (~) | Apto para TFLite/edge |
|---|---:|---:|---:|---|
| `efficientnet_b0` | 5.3 M | ~77.1 % | 16 MB | Sí (float16; INT8 aceptable) |
| `shufflenet_v2_x1_0` | 2.3 M | ~69.4 % | 5 MB | Sí (mobile-native) |
| `efficientnet_lite0` | 4.7 M | ~74.9 % | 14 MB | Sí (diseñado para INT8) |

Los tres parten de pesos pre-entrenados en ImageNet <sup>[[16]](#ref-16)</sup> y se ajustan sobre el dataset de maíz con:
- `CrossEntropyLoss` con pesos de clase inversamente proporcionales a la frecuencia <sup>[[12]](#ref-12)</sup>
- `WeightedRandomSampler` para igualar la frecuencia efectiva durante entrenamiento
- Pipeline de augmentation extendido para las 5 clases minoritarias

El modelo con mejor **macro-F1** en el conjunto de prueba establece el **umbral de referencia**: actualmente `efficientnet_b0`, con macro-F1 0.9146, que las arquitecturas posteriores (ResNet-50, ConvNeXt, ViT) deberán superar de forma consistente.

---

## Referencias

<a id="ref-6"></a>[6] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, y L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 4510–4520.

<a id="ref-8"></a>[8] M. Tan y Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, Long Beach, CA, USA, 2019, pp. 6105–6114.

<a id="ref-9"></a>[9] J. Hu, L. Shen, y G. Sun, "Squeeze-and-Excitation Networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 7132–7141.

<a id="ref-12"></a>[12] T.-Y. Lin, P. Goyal, R. Girshick, K. He, y P. Dollár, "Focal Loss for Dense Object Detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Venice, Italy, 2017, pp. 2980–2988.

<a id="ref-13"></a>[13] Google Brain / TensorFlow Team, "Higher Accuracy on Vision Models with EfficientNet-Lite," *TensorFlow Blog*, Mar. 2020. [Online]. Available: https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html

<a id="ref-16"></a>[16] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, y L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Miami, FL, USA, 2009, pp. 248–255.

<a id="ref-17"></a>[17] N. Ma, X. Zhang, H.-T. Zheng, y J. Sun, "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, Munich, Germany, 2018, pp. 116–131.
