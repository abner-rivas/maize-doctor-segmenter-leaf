# Preflight y control del entrenamiento cloud

El preflight local está diseñado como auditoría no destructiva. Recalcula locks, valida el dataset, inspecciona dependencias y hardware, revisa la compatibilidad del modelo candidato y rasteriza un batch de humo. También genera una configuración recomendada y el comando exacto, pero registra contadores de seguridad en cero para modelos construidos, forward, backward, pasos de optimizador, épocas, checkpoints, descargas e instalaciones.

En la ejecución documentada, el dataset y el loader aprobaron: se validaron 809/173/173 imágenes, 858/183/183 máscaras y 1,224 polígonos de una sola clase. El batch de humo reunió cuatro muestras de train, dos de validación y dos de test, con tensores finitos `[8,3,640,640]` y `[8,1,640,640]`. El estado local fue `blocked_by_missing_dependency` porque Ultralytics y los pesos iniciales no estaban instalados y CUDA no estaba disponible. Ese bloqueo no invalidó los datos; trasladó el entrenamiento a un entorno remoto autorizado.

## Paquete reproducible

El paquete cloud incluye únicamente los splits, `dataset.yaml`, cinco manifiestos necesarios, código, scripts y documentación mínima. Excluye el padre `all/`, fuentes externas, piloto, notebooks, entornos, cachés, outputs históricos y checkpoints. Un manifiesto separado describe el transporte del piloto y marca `included_in_training_package=false`.

El bootstrap fija `ultralytics==8.4.104` y `faster-coco-eval==1.7.2`, toma PyTorch y torchvision CUDA de la plataforma y ejecuta un `pip --dry-run`. Si el resolver intenta sustituir esas dos bibliotecas, el proceso se bloquea. El preflight remoto vuelve a verificar hashes, GPU, pesos y un forward sintético del head de segmentación.

Table: Gates explícitos del flujo remoto.

| Operación | Confirmación o condición | Resultado documentado |
|---|---|---|
| Bootstrap | Resolver compatible, sin reemplazar PyTorch | Aprobado |
| Preflight cloud | GPU, locks, pesos y forward | Aprobado |
| Smoke | `CONFIRM_SEGMENTATION_SMOKE_TRAINING=1` | Aprobado, batch 26 |
| Entrenamiento | `CONFIRM_SEGMENTATION_TRAINING=1` | C-01 y D-01 completados |
| Reanudación | Confirmación de entrenamiento y motivo | Disponible, no automática |
| Test | Split efectivo, conteos y checkpoint congelado | Bloqueado |
| Piloto externo | Test aprobado y pilot-gate | No ejecutado |

El uso de guards separados reduce el riesgo de que una comprobación inocua inicie épocas con coste. Los resultados se escriben en almacenamiento persistente y cada experimento tiene nombre propio. El runner exige que una corrida no sobrescriba otra y conserva configuración efectiva, checkpoints, curvas, resumen, entorno, duración y VRAM.

## Bloqueo de la evaluación final

Ultralytics 8.4.104 deduplica etiquetas por combinación de clase y bbox. En test existen dos polígonos diferentes que ocupan el marco completo y comparten esa clave; por eso 183 anotaciones raw se convierten en 182 instancias efectivas. El runner detecta la diferencia y bloquea el resumen, en vez de aceptar silenciosamente una población distinta. Resolver este contrato exige una decisión sobre datos o evaluación, no relajar la aserción para obtener un número.

