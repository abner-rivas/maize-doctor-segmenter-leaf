# Experimentos

Con la arquitectura ya acotada a unos pocos candidatos, los experimentos del pipeline principal dejarán de comparar modelos y pasarán a comparar decisiones de entrenamiento. 

El espacio de experimentos se reorientará hacia las decisiones introducidas en [preprocesado](./preprocessed) y [entrenamiento](./entrenamiento). Se compararán la estrategia de balanceo (sampler solo frente a sampler más pérdida ponderada, y con qué intensidad), el efecto de la regularización (label smoothing, mixup y cutmix) sobre el sobreajuste y la calibración, el schedule de learning rate (cosine frente a plateau, con y sin warmup), el fine-tuning en dos fases frente al completo, y el rendimiento con datos completos frente al régimen topado.
