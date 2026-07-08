# DoctorMaiz - Detección de Enfermedades, Plagas y Deficiencias Nutricionales en Cultivos de Maíz

> **Clasificación mediante Deep Learning en Dispositivos Móviles (Edge AI Offline)**

---

## Descripción

DoctorMaiz es un sistema de clasificación de enfermedades foliares, plagas y deficiencias nutricionales en cultivos de maíz orientado a pequeños agricultores de subsistencia en zonas rurales sin conectividad. Utiliza un modelo de Deep Learning cuantizado (TensorFlow Lite Int8) embebido en una aplicación Android que opera completamente offline.

### Problema

- El maíz representa una fuente crítica de alimentación en El Salvador, donde la agricultura aporta el **5.6% del PIB**
- El **82.1% de los productores** son pequeños agricultores con acceso limitado a asistencia técnica
- Las enfermedades, plagas y deficiencias nutricionales pueden destruir hasta el **70% de una cosecha**
- El diagnóstico actual depende de experiencia empírica y no de análisis técnico objetivo

### Solución

Una aplicación móvil que, dada una fotografía de hoja de maíz, identifica la enfermedad, plaga o deficiencia nutricional presente y orienta al agricultor sobre el tratamiento adecuado - sin necesidad de conexión a internet.

---

## Clases Objetivo

### Enfermedades y plagas foliares

| Clase | Patógeno/Agente | Síntomas | Lab | Real | Total |
|---|---|---|---:|---:|---:|
| Roya común *(Common Rust)* | *Puccinia sorghi* | Pústulas anaranjadas en ambas caras | 2 150 | 106 ⚠️ | 2 256 |
| Tizón foliar del norte *(NCLB)* | *Exserohilum turcicum* | Lesiones alargadas grisáceas | 888 | 5 942 | 6 830 |
| Mancha gris *(GLS)* | *Cercospora zeae-maydis* | Lesiones rectangulares grises | 513 | 606 | 1 119 |
| Necrosis letal *(MLN)* | Complejo viral (MCMV + potyvirus) | Rayado clorótico, necrosis progresiva y muerte de la planta | 0 | 6 415 | 6 415 |
| Hoja sana *(Healthy)* | - | Sin síntomas visibles | 0 | 8 744 | 8 744 |
| Gusano cogollero *(Fall Armyworm)* | *Spodoptera frugiperda* | Daño por masticación, excrementos en cogollo | 0 | 4 858 | 4 858 |

> `aphids_pest` (áfidos) se evaluó pero se descartó del alcance: solo ~77 imágenes disponibles, insuficientes para augmentation viable.

### Deficiencias nutricionales

| Clase | Síntomas | Lab | Real | Total |
|---|---|---:|---:|---:|
| Deficiencia de nitrógeno *(Nitrogen)* | Amarillamiento en "V" desde puntas de hojas inferiores | 0 | 523 ⚠️ | 523 |
| Deficiencia de fósforo *(Phosphorus)* | Bordes y puntas moradas/rojizas en hojas jóvenes | 0 | 612 ⚠️ | 612 |
| Deficiencia de potasio *(Potassium)* | Necrosis marginal en hojas más viejas | 0 | 266 ⚠️ | 266 |


---

## Objetivos Técnicos

| Métrica | Meta |
|---|---|
| Macro F1 Score | ≥ 0.85 |
| Tamaño del modelo (post Int8) | ≤ 20 MB |
| Latencia de inferencia | ≤ 300 ms/imagen |
| Dispositivo objetivo | Android ≥ 4 GB RAM, Snapdragon 6xx |
| Arquitectura base | En evaluación entre 6 baselines: EfficientNet-B0/Lite0, MobileNetV3-Large, FastViT-T8, GhostNetV2-100, ShuffleNetV2-x1.0 |

---

## Datasets

Se consolidaron **8 fuentes de datos públicas** para construir el corpus de entrenamiento:

| Dataset | Dominio | Imágenes (maíz) | Licencia |
|---|---|---|---|
| [Maize in Field](docs/es/datasets/maize-in-field-dataset.md) | Campo real (Sudáfrica) | ~2 223 | CC BY-NC-SA 4.0 |
| [Maize Diseases](docs/es/datasets/maize-diseases.md) | Lab + campo (PlantVillage v1.0/v1.1) | ~16 162 | CC BY-NC-SA 4.0 |
| [Corn Leaf Diseases](docs/es/datasets/corn-leaf-diseases.md) | Lab augmentado (×17) | 52 360 | MIT |
| [CropDG Unified Multidomain](docs/es/datasets/cropdg-unified-multidomain.md) | Multi-dominio | ~13 275 | CC BY-NC-SA 4.0 |
| [Maize, Beans & Tomatoes Africa](docs/es/datasets/maize-beans-tomatoes-africa.md) | Campo real (África) | 23 286 | Apache 2.0 + CC |
| [Multicrop Disease - Maize Pests and Disease](docs/es/datasets/multicrop-disease-maiz-disease-pests-and-disease.md) | Mixto | - | Desconocida |
| [Maize Nutrient Deficiency](docs/es/datasets/maize-nutrient-deficiency.md) | Campo real (India) | 463 | CC BY 4.0 |
| [Corn Leaf - Roboflow](docs/es/datasets/corn-leaf-roboflow.md) | Campo real | 3 943 | CC BY 4.0 |


### Estrategia de Augmentation

El dataset *Corn Leaf Diseases* aplica 17 técnicas de augmentation documentadas:

`brightness_adjusted` · `contrast_adjusted` · `cropped` · `flipped_horizontal` · `flipped_vertical` · `gaussian_noise` · `high_pass` · `hist_equalized` · `jittered` · `laplacian` · `poisson_noise` · `rotated` · `salt_pepper_noise` · `saturation_adjusted` · `sobel` · `translated` · `unsharp_mask`

---

## Metodología

El proyecto sigue el marco **CRISP-DM iterativo**:

1. **Comprensión del negocio** - Definición del problema agrícola y restricciones de despliegue
2. **Comprensión de datos** - Consolidación y auditoría de 8 fuentes públicas
3. **Preparación** - Limpieza, estandarización (224×224 px), deduplicación, augmentation
4. **Modelado** - Fine-tuning de 6 arquitecturas baseline para comparar rápido y barato
5. **Evaluación** - Macro F1 ≥ 0.85 en conjunto independiente de imágenes de campo real
6. **Despliegue** - PWA offline con TFLite Int8 + sincronización opcional

---

## Pipeline de Machine Learning

El código vive en `src/` (librería instalable, `pip install -e .`) y `scripts/` (entrypoints).
Dos pipelines paralelos sobre el mismo dataset limpio (`clean/`):

- **Baselines** (`scripts/pipeline/train_baselines.py`): 6 arquitecturas pre-entrenadas
  (EfficientNet-B0/Lite0, MobileNetV3-Large, FastViT-T8, GhostNetV2-100, ShuffleNetV2-x1.0),
  funcional de punta a punta. Por defecto entrena sobre un subset configurable
  (`config/dataset.yaml -> baseline:`, 4 clases y hasta 500 imágenes por clase) para comparar
  arquitecturas rápido y barato — ver [Baselines](docs/es/baselines/index.md).
- **Pipeline principal** (`scripts/pipeline/train.py`): comparte toda la infraestructura de
  datos y modelos con baselines; el loop de entrenamiento está pendiente de implementar.

Guía de instalación local (venv, `.env`, dataset) en [LOCAL.md](LOCAL.md).

---

## Comandos

Todos los comandos usan `make` (detecta Windows/Linux automáticamente). Variables comunes:
`MODELS` (nombre o "all"), `EPOCHS`, `NO_CAP=1` / `MAX_PER_CLASS=<n>` (override del tope de
imágenes por clase del perfil baseline), `RUN` (run_id específico), `SAMPLE_SIZE`.

### Locales

```bash
make install                        # pip install -e ".[dev,analysis,xai,cloud]"
make download-dataset                # clean/ (HF Hub, fallback Google Drive)

make splits                          # splits completos (9 clases) -> outputs/splits/seed_42/
make splits-baseline [NO_CAP=1 | MAX_PER_CLASS=<n>]   # perfil baseline -> outputs/splits/seed_42_baseline/

make train-baselines [MODELS=<nombre>] [NO_CAP=1 | MAX_PER_CLASS=<n>]   # genera splits (lazy) y entrena
make train                           # pipeline principal (loop de entrenamiento pendiente)

make explain-lime [MODELS=<nombre> RUN=<id> IMAGE=<ruta> OUTPUT=<ruta>]   # reporte visual LIME+Grad-CAM
make explain-report [MODELS=<nombre> RUN=<id> SAMPLE_SIZE=<n> NUM_SAMPLES=<n>]  # fidelidad agregada
make explain-errors [MODELS=<nombre> RUN=<id> NUM_SAMPLES=<n>]   # LIME dirigido a errores

make clean-outputs                   # borra outputs/ (splits, runs, reportes — todo regenerable)
make summary / make test-loader / make lint / make fmt
```

### Modal (GPU en la nube)

Misma CLI que los comandos locales — cualquier combinación de banderas que funcione en local
funciona igual en Modal. Detalle completo en [docs/es/deployment/modal.md](docs/es/deployment/modal.md).

```bash
make modal-seed                                            # sube clean/ al Volume (una vez)
make modal-train-baselines [MODELS=<nombre>] [NO_CAP=1 | MAX_PER_CLASS=<n>]
make modal-explain-lime / modal-explain-report / modal-explain-errors [MODELS=<nombre> RUN=<id>]
make modal-clean-outputs                                    # vacía el Volume corn-outputs
make modal-pull                                             # trae outputs-remote/ con runs + reportes
```

Para GPU alquilada por SSH en [vast.ai](https://vast.ai) en vez de Modal, ver
[docs/es/deployment/vast-ai.md](docs/es/deployment/vast-ai.md).

---

## Equipo

| Nombre | Carné |
|---|---|
| Josias Abner Rivas Fuentes | RF20010 |
| David Alejandro Deras Cerros | DC19019 |
| Elmer Edenilson Rosales Molina | RM20001 |

---

## Estructura del Proyecto

```
corn-leaf-desease-project/
├── config/dataset.yaml   # Clases, tamaño de imagen, seed, perfil "baseline"
├── docs/es/              # Documentación (VitePress): datasets, EDA, baselines, deployment
├── notebooks/            # Análisis exploratorio
├── scripts/
│   ├── dataset/          # Subida/descarga de clean/ (Hugging Face Hub, Google Drive)
│   ├── pipeline/         # create_splits.py, train_baselines.py, train.py, explain_lime.py, explain_report.py
│   ├── modal/            # Entrenamiento/explicabilidad en GPU de Modal
│   └── vastai/           # Orquestación de GPU remota en vast.ai
├── src/                  # Librería principal (pip install -e .)
│   ├── data/             # CornDataset, loader, splitter, transforms
│   ├── explainability/   # LIME + Grad-CAM (post-hoc, no acoplado al entrenamiento)
│   └── models/           # Registro de modelos + 6 baselines
├── Dockerfile            # Imagen reproducible (Python 3.11 + PyTorch CUDA) para GPU remota
├── Makefile
└── pyproject.toml
```

Documentación completa construida con VitePress (`npm install && npm run docs:dev`, disponible
en `http://localhost:5173`).

---

## Estado del Proyecto

- [x] Documentación de datasets consolidados (8 fuentes, 9 clases)
- [x] Scripts de limpieza y organización de datos en `data/clean/`
- [x] Análisis exploratorio de datos (EDA)
- [x] Pipeline de preparación de datos (splits estratificados, perfil baseline configurable)
- [x] Data augmentation para clases minoritarias (pipeline extendido por clase en `transforms.py`)
- [x] Entrenamiento de 6 baselines (EfficientNet-B0/Lite0, MobileNetV3-Large, FastViT-T8, GhostNetV2-100, ShuffleNetV2-x1.0) + soporte GPU remota (vast.ai, Modal)
- [x] Explicabilidad post-hoc (LIME + Grad-CAM, análisis de errores y fidelidad agregada)
- [ ] Loop de entrenamiento del pipeline principal (`scripts/pipeline/train.py`)
- [ ] Evaluación exhaustiva y selección de modelo final
- [ ] Aplicación Android con TensorFlow Lite

---

## Licencia

Este proyecto es de carácter académico. Los datasets utilizados mantienen sus licencias originales (ver documentación individual de cada dataset).
