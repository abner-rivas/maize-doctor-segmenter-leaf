# Trabajos previos

El diagnóstico automatizado de enfermedades foliares en maíz (*Zea mays L.*) ha dado grandes pasos desde la inspección visual en los campos, hacia sistemas de visión por computadora basados en aprendizaje profundo. Haremos un resumen de los trabajos previos de este campo, para después justificar por qué el alcance de este proyecto se concentra en la clasificación y cómo lo hemos propuesto como un paso intermedio hacia un sistema completo de segmentación y clasificación.

Los primeros proyectos se apoyaban en el aprendizaje automático clásico: SVM, K-Nearest Neighbors o Random Forest sobre características diseñadas a mano, como descriptores de color, textura y forma de la lesión. Ese enfoque alcanzaba precisiones del orden del 79 % al 90 % <sup>[[1]](#ref-1)</sup>. La transición hacia las Redes Neuronales Convolucionales cambió la lógica del problema, porque la red aprende por sí misma la jerarquía de rasgos relevantes, junto con ese salto se consolidó también el uso del F1-score como métrica principal en lugar del accuracy, dado que por la naturaleza de estos conjuntos de datos, comúnmente no se dispone de un número equilibrado de imágenes por clase.

## El entorno de las fotos

Dentro de toda la investigación realizada, el problema más crítico que se ha documentado es la brecha de dominio entre laboratorio y campo. La mayoría de los modelos pioneros se entrenaron con *PlantVillage*, un conjunto que captura hojas aisladas sobre fondos homogéneos y con iluminación controlada, donde las CNN alcanzan con regularidad exactitudes de entre 95.81 % y 99.53 % <sup>[[2]](#ref-2)</sup>, pero aún con esas cifras, cuando esos mismos modelos se evalúan contra conjuntos capturados en campo real, como *PlantDoc*, *FieldPlant* o *CD&S*, la precisión llega a desplomarse a rangos críticos de entre 33.27 % y 39.87 % <sup>[[2]](#ref-2)</sup>.

Ya existen muchas investigaciones que han llegado a los mismos análisis y conclusiones, y es que en el campo tenemos hojas enfermas que se superponen con hojas sanas, malezas, suelo y otras estructuras de la planta, además de sombras, reflejos solares y una variabilidad en la luz que afecta la visibilidad de los rasgos. 

Para intentar combatir esos problemas, algunos trabajos han recurrido a aumento de datos agresivo (data augmentation), a mezclar imágenes de laboratorio y campo en el entrenamiento y a técnicas de eliminación de fondo mediante canales RGBA. Con esas correcciones, arquitecturas como DenseNet169 logran estabilizar la precisión de generalización cruzada en un rango del 77.50 % al 81.60 % bajo condiciones reales <sup>[[3]](#ref-3)</sup>.

## Tecnologías y limitaciones actuales

Investigaciones actuales cambian la arquitectura de la solución, de CNN puras hacia esquemas híbridos y en cascada que separan el problema para intenar lidiar con los problemas que mencionamos arriba.

Uno de los esquemas actuales combina Redes Neuronales Convolucionales con Vision Transformers: las CNN extraen las texturas finas del patógeno, mientras que el Vision Transformer modela mediante autoatención las relaciones de largo alcance entre lesiones separadas dentro de la misma hoja. Trabajos como el híbrido CNN-ViT de Shandilya et al. reportan un F1-score cercano al 99.13 % sobre conjuntos mixtos <sup>[[4]](#ref-4)</sup>, y sistemas desplegados en campo como ZeaWatch alcanzan métricas de validación casi perfectas, aunque su confianza de predicción documentada cae al 70.8 % una vez instalado en los lugares a utilizar <sup>[[5]](#ref-5)</sup>, lo que de nuevo nos trae a la brecha de dominio lab / real.

Un segundo esquema es abordar el problema de dominio en 2 pasos:

- Un primer módulo de detección, del tipo Faster R-CNN o LS-RCNN, **AÍSLA** la hoja del entorno.
- Para que luego un segundo módulo clasifique la enfermedad.

Con esto lo que se evita es que el clasificador aprenda a asociar el fondo con el diagnóstico de la enfermedad <sup>[[6]](#ref-6)</sup>.

La misma idea aparece en otros enfoques que integran cámaras de profundidad RGB-D para recortar físicamente el fondo mediante el mapa de profundidad, después de eso un modelo ligero como MobileNetV2 clasifica la hoja ya segmentada <sup>[[7]](#ref-7)</sup>.

Otros problemas también identificados en otros trabajos son:

- La precisión artificialmente inflada de los modelos entrenados sobre imágenes sin recortar, que aprenden a reconocer el color de un suelo o una sombra en lugar del síntoma. 
- La dificultad con detectar las enfermedades en fases tempranas, por ejemplo cuando los síntomas del tizón foliar del norte o de la mancha gris son pequeños y difíciles de distinguir.
- Relacionado al punto anterior, también existe confusión entre clases y la presencia de múltiples infecciones.

Por último un problema con el que también se debe lidiar en este proyecto es el trade-off entre precisión y despliegue, porque los modelos de última generación que logran métricas de validación cercanas al 99 % son demasiado pesados para ejecutarse en un teléfono de gama media sin conexión, y por lo tanto no son prácticos para una solución de inferencia local.

## Alcance de este proyecto

Bajo esta investigación se ha podido constatar que un sistema robusto para el diagnóstico de enfermedades foliares debería, idealmente, buscar una solución en 2 etapas: primero aislar de alguna forma la hoja del fondo complejo, y después clasificar la enfermedad sobre esa región ya limpia.

El alcance de este proyecto se ideó para abordar la segunda etapa, la clasificación, con la idea de que en un futuro cercano se pueda integrar un módulo de segmentación.

Es en esta etapa de clasificación donde reside el "juicio" agronómico del sistema, se priorizará la evaluación sobre imágenes de campo real y no solo de laboratorio para no caer en esa ilusión de fiabilidad, se adopta el F1-macro con umbral de 0.85 como criterio de viabilidad para que las clases minoritarias pesen igual que las mayoritarias, se eligen arquitecturas ligeras como las del tipo EfficientNet-B0 pensando en el despliegue TFLite offline con un modelo muy liviano y una latencia por debajo de 300 ms.

El clasificador que se trabaja en este proyecto está diseñado para operar sobre la hoja como región de interés, de modo que el día en que se anteponga un módulo de segmentación, ya sea una red en cascada o un recorte por profundidad, este componente o pipeline pueda recibir la hoja aislada sin la necesidad de rediseñarse en gran medida.

En ese sentido la propuesta de este trabajo es ser un producto útil por sí mismo, capaz de dar un diagnóstico offline a un pequeño productor, y a la misma vez la segunda mitad ya construida de un sistema futuro que segmente, clasifique y de mejores diagnósticos.

## Referencias

<a id="ref-1"></a>[1] Autores varios, "Research Advances in Maize Crop Disease Detection Using Machine Learning and Deep Learning Approaches," *Computers*, vol. 15, no. 2, art. 99, 2026. [Online]. Available: https://www.mdpi.com/2073-431X/15/2/99

<a id="ref-2"></a>[2] A. Ahmad, A. El Gamal, y D. Saraswat, "Toward Generalization of Deep Learning-Based Plant Disease Identification Under Controlled and Field Conditions," *IEEE Access*, vol. 11, pp. 9042–9057, 2023.

<a id="ref-3"></a>[3] A. Ahmad, A. El Gamal, y D. Saraswat, "Towards Generalization of Deep Learning-Based Plant Disease Identification Under Controlled and Field Conditions," Purdue e-Pubs, 2023. [Online]. Available: https://docs.lib.purdue.edu/fund/58/

<a id="ref-4"></a>[4] Shandilya et al., "Enhanced Maize Leaf Disease Detection and Classification Using an Integrated CNN-ViT Model," *Food Science & Nutrition*, 2025. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC12208913/

<a id="ref-5"></a>[5] J. Baya et al., "A Mobile-based system for maize leaf disease detection and classification using machine learning algorithms and deep learning: A Case Study of ZeaWatch," Preprint, 2024.

<a id="ref-6"></a>[6] H. Liu, H. Lv, J. Li, Y. Liu, y L. Deng, "Research on Maize Disease Identification Methods in Complex Environments Based on Cascade Networks and Two-Stage Transfer Learning," *Scientific Reports*, vol. 12, no. 1, art. 18914, 2022.

<a id="ref-7"></a>[7] F. Nan et al., "A novel method for maize leaf disease classification using the RGB-D post-segmentation image data," *Frontiers in Plant Science*, vol. 14, art. 1268015, 2023.

<a id="ref-8"></a>[8] Wu et al., "AdaptiveLeaf: Lightweight Multi-Scale Framework for Small-Target Detection of Maize Leaf Diseases," vol. 16, no. 13, art. 1415, 2026.
