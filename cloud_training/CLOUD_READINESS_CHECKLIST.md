# Checklist cloud del segmentador

## Dataset

- [ ] `dataset_lock.json` y `split_lock.json` aprobados.
- [ ] Fingerprints parent/train/val/test recalculados y coincidentes.
- [ ] Cero fugas entre splits y contra el piloto.
- [ ] `dataset.yaml` portable con clase única `maize_leaf`.

## Entorno

- [ ] Python compatible y PyTorch/torchvision CUDA provistos por la plataforma.
- [ ] GPU visible con al menos 12 GiB de VRAM.
- [ ] `ultralytics==8.4.104` y `faster-coco-eval==1.7.2` fijados.
- [ ] El dry-run de pip no reemplaza torch ni torchvision.

## Ejecución

- [ ] Preflight de GPU, pesos y forward aprobado.
- [ ] Smoke autorizado, finito y con batch resuelto registrado.
- [ ] Configuración final congelada.
- [ ] Entrenamiento o resume autorizados explícitamente.
- [ ] Checkpoints y summaries persistidos fuera del filesystem efímero.

## Evaluación

- [ ] Conteos efectivos de imágenes e instancias coinciden con el contrato.
- [ ] Test interno produce métricas y fingerprint del checkpoint.
- [ ] Métricas downstream revisadas por fuente y tamaño de máscara.
- [ ] Piloto externo ejecutado únicamente después del test interno.
- [ ] Quality gate auditado con casos humanos y paneles visuales.
