# Entrenamiento

El pipeline principal compartirá la infraestructura de datos y modelos del proyecto (registro de arquitecturas, dataloaders) y cambiará el loop de entrenamiento para exprimir el rendimiento de la arquitectura ya elegida, incorporando los mecanismos de optimización y regularización que se describen a continuación.

## Learning rate scheduler

El learning rate dejará de ser constante y se adaptará a la fase del entrenamiento. Se contempla `cosine annealing`, que lo decae de forma suave hasta casi cero y suele generalizar bien en fine-tuning de visión, o `ReduceLROnPlateau`, que lo baja solo cuando la validación deja de mejorar, con un `warmup` inicial de pocos epochs para las arquitecturas que arrancan lento. Un lr fijo mantiene pasos demasiado grandes en los epochs finales, cuando el modelo debería estar afinando, y contribuye al sobreajuste.

## Early stopping

Si la métrica de validación no mejora durante un número de epochs (`patience` del orden de 5 a 8), la corrida se detendrá. Como se conservará el mejor checkpoint (`best.pth`), esto no afectará la calidad del modelo final pero recortará el cómputo malgastado tras el mejor epoch y abaratará las barridas de hiperparámetros.

## Regularización

Se añadirán varias capas complementarias de regularización. El `label smoothing` (típicamente 0.1) evita que el modelo se vuelva sobreconfiado forzando probabilidades exactamente 1/0, lo que mejora la calibración de las probabilidades, un punto crítico para el caso de uso real. A esto se sumarán el `weight decay` de AdamW.

## Configuración prevista

| Elemento | Configuración |
|---|---|
| Datos | 9 clases, 100% de imágenes |
| Learning rate | Scheduler (cosine o plateau) con warmup |
| Parada | Early stopping (`patience` 5 a 8) con `best.pth` |
| Pérdida | CrossEntropy ponderada con label smoothing |
| Balanceo | Sampler ponderado más pérdida ponderada |
| Optimizador | AdamW con gradient clipping |
