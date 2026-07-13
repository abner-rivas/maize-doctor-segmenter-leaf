# Preprocesado

El pipeline principal hereda el preprocesado base del proyecto (normalizado con estadísticas de ImageNet, corrección EXIF, RGB estricto, división estratificada por `label + environment` y seed fija 42), documentado en [Preprocesado](../preprocessed/index.md).

Pero para este pipeline se añaden varias decisiones de preprocesado:

Lo primero es que ahora si se entrenerá sobre el 100% de las imágenes válidas de las 9 clases, sin límite por clase, una vez elegida la arquitectura ya no interesa tener corridas más rápidas y baratas, sino aprovechar cada imagen disponible. Con todas las imágenes en juego, califican como minoritarias (más de 4x por debajo de la clase mayor) `gray_leaf_spot`, `nitrogen_deficiency`, `phosphorus_deficiency` y `potassium_deficiency`, de modo que las tres deficiencias nutricionales reciben tanto el muestreo ponderado como la augmentación agresiva.

## Balanceo de dos capas

El desbalance se atacará con dos mecanismos complementarios. El `WeightedRandomSampler` igualará la frecuencia efectiva de cada clase por epoch asignando a cada muestra un peso `1 / count_of_its_class`, y como la augmentation es en caliente cada repetición de una imagen minoritaria se verá distinta. A esto se sumará una pérdida `CrossEntropyLoss` ponderada por clase, que penalizará más los errores sobre las clases escasas.

## Validación del dominio

El preprocesado preparará cortes cruzados por entorno (entrenar en un dominio y evaluar en el otro), que permitirán medir en [evaluación](./evaluacion) si el modelo generaliza entre laboratorio y campo o se apoya en el fondo.
