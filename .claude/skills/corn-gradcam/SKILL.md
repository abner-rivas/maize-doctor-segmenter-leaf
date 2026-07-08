---
name: corn-gradcam
description: Use when editing src/explainability/gradcam.py, adding a new model to MODEL_REGISTRY, or asked about the Grad-CAM panel in explainability reports for the corn leaf disease project.
---

# Corn Grad-CAM

`src/explainability/gradcam.py` añade un 4to panel opcional (`config/dataset.yaml -> gradcam.enabled`) vía hooks nativos de PyTorch, sin dependencias nuevas.

`GRADCAM_TARGET_LAYERS` mapea cada modelo del registry a su última capa con salida espacial.

**Al añadir un modelo nuevo a `MODEL_REGISTRY`, agregar también su entrada en `GRADCAM_TARGET_LAYERS`** - si no, Grad-CAM se omite con warning (fallback a 3 paneles).
