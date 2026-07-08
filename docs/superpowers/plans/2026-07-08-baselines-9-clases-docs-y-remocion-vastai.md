# Baselines 9 clases + remoción de vast.ai — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar como baseline canónico y documentado únicamente la última corrida de 9 clases (cap 1500) con 3 modelos (efficientnet_b0, shufflenet_v2_x1_0, efficientnet_lite0), eliminar todo lo relacionado a vast.ai, y confinar la exploración de 4 clases / 8 arquitecturas a una sola página de "Experimentos".

**Architecture:** Cambios de documentación (VitePress bajo `docs/es/`) + limpieza de orquestación (`scripts/vastai/`, `Dockerfile`). No se toca código de `src/` ni el registro de modelos (siguen registradas las 8 arquitecturas; solo cambia lo que la doc presenta como baseline). Todo en la rama actual `feat/modal-baselines`.

**Tech Stack:** Markdown (VitePress), TypeScript (`config.mts`), Makefile, Docker.

## Global Constraints

Estos son los **hechos canónicos**. Toda página de docs (salvo `experimentos.md`) debe reflejarlos y **no** debe mencionar 4 clases, cap 500, ni las corridas exploratorias.

- **Baseline canónico = 1 sola corrida:** 9 clases (todas las de `dataset.classes`), `baseline.max_images_per_class: 1500` ("cap la cabeza, conserva la cola"), splits `outputs/splits/seed_42_baseline/`, estratificado por `label + environment`, seed 42.
- **Volumen:** ~10 020 imágenes tras el cap; test N = 1503 (15 %).
- **Config de entrenamiento:** 30 épocas, `AdamW` (lr 1e-4, wd 1e-4), `CrossEntropyLoss` ponderada por clase, `WeightedRandomSampler`, augmentation minority en caliente, backbones pre-entrenados en ImageNet, `image_size` por-modelo (los 3 baselines canónicos usan 224×224).
- **Los 3 baselines canónicos y sus métricas de test (9 clases):**
  | Modelo | Accuracy | macro-F1 |
  |---|---|---|
  | `efficientnet_b0` | 0.9521 | **0.9146** |
  | `shufflenet_v2_x1_0` | 0.9508 | 0.9030 |
  | `efficientnet_lite0` | 0.9474 | 0.8951 |
- **Cuello de botella conocido:** `potassium_deficiency` (f1 0.49–0.62 según modelo; 266 imágenes totales). Clúster de confusión de deficiencias N/P/K y clúster de lesiones foliares `gray_leaf_spot`/`northern_corn_leaf_blight`.
- **Solo en `experimentos.md`:** las 2 corridas de 4 clases (cap 500 y sin cap) exploraron hasta **8 arquitecturas**: `efficientnet_b0`, `efficientnet_b4`, `efficientnet_lite0`, `fastvit_t8`, `ghostnetv2_100`, `mobilenet_v3_large`, `mobilenet_v3_small`, `shufflenet_v2_x1_0`. Resultados completos → enlace a un ZIP en Google Drive (pendiente de subir).
- **No tocar:** `src/`, el registro de modelos, el árbol `docs/es/pipeline/*` (Principal, comentado en nav), la corrida no-cap/500 en disco.

---

## Archivos afectados

| Archivo | Acción |
|---|---|
| `scripts/vastai/` (dir completo) | **Eliminar** |
| `docs/es/deployment/vast-ai.md` | **Eliminar** |
| `Dockerfile`, `.dockerignore` | **Eliminar** (vast-específicos; Modal usa su propia imagen) |
| `docs/.vitepress/config.mts` | Modificar (nav Deployment → Modal; sidebar Baselines + "Modelos") |
| `README.md` | Modificar (lista de baselines, refs vast, árbol, estado) |
| `docs/es/deployment/modal.md` | Modificar (quitar cross-ref a vast) |
| `.env.example` | Modificar (quitar línea vast) |
| `.claude/agents/baseline-ml.md` | Modificar (quitar ref vast si aplica) |
| `docs/es/baselines/index.md` | Reescribir (swap mobilenet→shufflenet, 9 clases/1500) |
| `docs/es/pipeline-baselines/preprocesado.md` | Reescribir (nota baseline 9cls/1500 + link a compartido) |
| `docs/es/pipeline-baselines/entrenamiento.md` | Escribir (stub → contenido) |
| `docs/es/pipeline-baselines/evaluacion.md` | Escribir (stub → contenido) |
| `docs/es/pipeline-baselines/interpretabilidad.md` | Escribir (stub → contenido) |
| `docs/es/pipeline-baselines/experimentos.md` | Escribir (stub → contenido, ÚNICO lugar con la exploración) |
| `docs/es/preprocesado/index.md` | Modificar (fix "8 clases"→"9 clases" + nota cap 1500) |

---

## Fase A — Remoción de vast.ai

### Task A1: Eliminar código y doc de orquestación vast.ai

**Files:**
- Delete: `scripts/vastai/` (incluye `launch.py`, `onstart.sh`, `__pycache__/`)
- Delete: `docs/es/deployment/vast-ai.md`
- Delete: `Dockerfile`, `.dockerignore`

**Interfaces:**
- Produces: nada. (Ningún módulo de `src/` importa `scripts/vastai`; Modal define su imagen en `scripts/modal/_common.py`, no usa el `Dockerfile` de raíz.)

- [ ] **Step 1: Verificar que nada del código importe vastai ni el Dockerfile de raíz**

Run: `grep -rniE "vastai|scripts.vastai|from Dockerfile" src scripts/modal scripts/pipeline 2>/dev/null | grep -v venv`
Expected: sin resultados (el Dockerfile solo lo referencia la doc vast, que también se borra).

- [ ] **Step 2: Eliminar los archivos**

```bash
git rm -r scripts/vastai docs/es/deployment/vast-ai.md Dockerfile .dockerignore
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove vast.ai orchestration (scripts/vastai, Dockerfile, deployment doc)"
```

> Nota para el revisor: `Dockerfile`/`.dockerignore` se eliminan porque su cabecera declara uso exclusivo para GPU remota estilo vast.ai y Modal no los usa. Si se quisiera conservar una imagen reproducible genérica, omitir su borrado en el Step 2 — es la única decisión con margen en esta tarea.

### Task A2: Limpiar referencias a vast.ai en navegación y prosa

**Files:**
- Modify: `docs/.vitepress/config.mts` (bloque `Deployment` en `esDatasetSidebar` y en `themeConfig.nav`)
- Modify: `docs/es/deployment/modal.md:4-6`
- Modify: `.env.example:9`
- Modify: `.claude/agents/baseline-ml.md` (si contiene ref a vast)

**Interfaces:**
- Consumes: la existencia de `docs/es/deployment/modal.md` (ya existe).
- Produces: la sección Deployment de la nav queda apuntando a Modal.

- [ ] **Step 1: Reemplazar la entrada Deployment del sidebar en `config.mts`**

Buscar (aparece 1 vez, en `esDatasetSidebar`):

```ts
  {
    text: "Deployment",
    items: [
      { text: "GPU en vast.ai", link: "/es/deployment/vast-ai" },
    ],
  },
```

Reemplazar por:

```ts
  {
    text: "Deployment",
    items: [
      { text: "GPU en Modal", link: "/es/deployment/modal" },
    ],
  },
```

- [ ] **Step 2: Reemplazar la entrada Deployment del `nav` en `config.mts`**

Buscar (aparece 1 vez, dentro de `themeConfig.nav`):

```ts
          {
            text: "Deployment",
            items: [
              { text: "GPU en vast.ai", link: "/es/deployment/vast-ai" },
            ],
          },
```

Reemplazar por:

```ts
          {
            text: "Deployment",
            items: [
              { text: "GPU en Modal", link: "/es/deployment/modal" },
            ],
          },
```

- [ ] **Step 3: Quitar el cross-ref a vast en `modal.md`**

Buscar:

```md
[Modal](https://modal.com/docs/guide). Modal coexiste con vast.ai (ver `vast-ai.md`); no lo
reemplaza. A diferencia de vast.ai (VM + SSH), en Modal defines código que corre en la nube y
se cobra por segundo, con auto-teardown (no hay que acordarse de destruir instancias).
```

Reemplazar por:

```md
[Modal](https://modal.com/docs/guide). En Modal defines código que corre en la nube y se cobra
por segundo, con auto-teardown (no hay que acordarse de destruir instancias).
```

- [ ] **Step 4: Quitar la línea vast de `.env.example`**

Buscar `#   Instancia vast.ai: DATASET_ROOT=/workspace/data` y eliminar esa línea.

- [ ] **Step 5: Revisar y limpiar `.claude/agents/baseline-ml.md`**

Run: `grep -n "vast" .claude/agents/baseline-ml.md`
Si hay una mención a vast.ai como entorno, reescribirla para nombrar solo Modal / local. Si es una referencia incidental, ajustarla en la misma línea.

- [ ] **Step 6: Verificar que no quedan referencias vivas a vast**

Run: `grep -rniE "vast" . --include="*.md" --include="*.mts" --include="*.py" --include="*.toml" --include="*.example" 2>/dev/null | grep -v venv | grep -v "/dist/" | grep -v ".superpowers/" | grep -v "docs/superpowers/plans/"`
Expected: sin resultados.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "docs: drop vast.ai from nav, modal doc, env example and agent guide"
```

---

## Fase B — Baselines: 3 modelos canónicos (9 clases)

### Task B1: Actualizar `docs/es/baselines/index.md` (swap modelo + datos 9 clases)

**Files:**
- Modify: `docs/es/baselines/index.md`

**Interfaces:**
- Produces: página de overview con exactamente 3 modelos: `efficientnet_b0`, `shufflenet_v2_x1_0`, `efficientnet_lite0`.

- [ ] **Step 1: Reemplazar la sección "## Dataset utilizado" completa**

Reemplazar el bloque que empieza en `## Dataset utilizado` y termina antes de `## Modelos seleccionados` por:

```md
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
```

- [ ] **Step 2: Actualizar el párrafo intro de los 3 modelos**

Buscar el párrafo que empieza con `Los tres modelos elegidos son redes convolucionales ligeras` y reemplazar su enumeración de modelos para que nombre **EfficientNet-B0, ShuffleNetV2-x1.0 y EfficientNet-Lite0** (cubren el eje precisión↔eficiencia y los tres convierten a TFLite para despliegue móvil offline).

- [ ] **Step 3: Reemplazar la subsección "### MobileNetV3-Large" por "### ShuffleNetV2-x1.0"**

Eliminar la subsección completa `### MobileNetV3-Large` (desde ese encabezado hasta antes de `### EfficientNet-B0`) e insertar en su lugar:

```md
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
```

- [ ] **Step 4: Reemplazar la tabla "## Comparación de los tres modelos"**

Reemplazar la tabla comparativa por:

```md
| Modelo | Parámetros | Top-1 ImageNet | Tamaño (~) | Apto para TFLite/edge |
|---|---:|---:|---:|---|
| `efficientnet_b0` | 5.3 M | ~77.1 % | 16 MB | Sí (float16; INT8 aceptable) |
| `shufflenet_v2_x1_0` | 2.3 M | ~69.4 % | 5 MB | Sí (mobile-native) |
| `efficientnet_lite0` | 4.7 M | ~74.9 % | 14 MB | Sí (diseñado para INT8) |
```

- [ ] **Step 5: Actualizar el párrafo de cierre de la comparación**

En el párrafo posterior a la tabla, mantener la descripción de `CrossEntropyLoss` ponderada + `WeightedRandomSampler` + augmentation minority, y cambiar la frase final para decir que el modelo con mejor **macro-F1** en test (actualmente `efficientnet_b0`, 0.9146) establece el umbral de referencia.

- [ ] **Step 6: Añadir la referencia [17] de ShuffleNetV2**

Al final de la sección `## Referencias`, añadir:

```md
<a id="ref-17"></a>[17] N. Ma, X. Zhang, H.-T. Zheng, y J. Sun, "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, Munich, Germany, 2018, pp. 116–131.
```

- [ ] **Step 7: Verificar que no queda mobilenet ni datos de 4 clases/500**

Run: `grep -niE "mobilenet|4 clases|500|2 000|nitrogen_deficiency.*fall_armyworm" docs/es/baselines/index.md`
Expected: sin resultados (salvo, si acaso, menciones en `experimentos`, que es otro archivo).

- [ ] **Step 8: Commit**

```bash
git add docs/es/baselines/index.md
git commit -m "docs(baselines): 3 baselines canonicos (b0/shufflenet/lite0) sobre 9 clases cap 1500"
```

---

## Fase C — Páginas del pipeline Baselines

### Task C1: `pipeline-baselines/preprocesado.md`

**Files:**
- Modify: `docs/es/pipeline-baselines/preprocesado.md`

- [ ] **Step 1: Reemplazar el contenido completo**

```md
# Preprocesado

El preprocesado (normalizado, división estratificada, balanceo y data augmentation) es
compartido entre todos los pipelines. Ver la documentación completa en
[Preprocesado](../preprocesado/index.md).

## Especificidad del baseline

El baseline consume el split `outputs/splits/seed_42_baseline/`, generado con
`make splits-baseline`. A diferencia del split completo (`seed_42`), aplica un **cap de 1 500
imágenes por clase** sobre las 9 clases: recorta solo las clases mayoritarias y conserva íntegras
las minoritarias (potasio 266, nitrógeno 523, fósforo 612). Todo lo demás —normalización
ImageNet, estratificación por `label + environment`, seed 42, sampler y augmentation— es idéntico
al pipeline principal.
```

- [ ] **Step 2: Commit**

```bash
git add docs/es/pipeline-baselines/preprocesado.md
git commit -m "docs(baselines): preprocesado remite al compartido + nota del cap 1500"
```

### Task C2: `pipeline-baselines/entrenamiento.md`

**Files:**
- Modify: `docs/es/pipeline-baselines/entrenamiento.md`

- [ ] **Step 1: Reemplazar el contenido completo**

```md
# Entrenamiento

Los baselines se entrenan con `scripts/pipeline/train_baselines.py`, que comparte toda la
infraestructura de datos y modelos con el pipeline principal. Cada arquitectura se construye
desde `MODEL_REGISTRY`, con backbone pre-entrenado en ImageNet y solo la capa de clasificación
reemplazada por una lineal de 9 salidas.

## Configuración canónica

| Hiperparámetro | Valor |
|---|---|
| Clases | 9 (perfil `baseline`, cap 1 500/clase) |
| Épocas | 30 |
| Optimizador | AdamW (lr 1e-4, weight decay 1e-4) |
| Pérdida | CrossEntropyLoss ponderada por clase |
| Balanceo | WeightedRandomSampler + augmentation minority en caliente |
| Pesos iniciales | ImageNet (`pretrained=True`) |
| `image_size` | por-modelo (los 3 baselines usan 224×224) |
| Batch size | 32 (auto-escalado según resolución del modelo) |

Cada corrida se versiona en `outputs/baselines/<modelo>/<run_id>/` con `best.pth`, `summary.json`,
`predictions.csv`, `train_history.csv` y los reportes de test. El `summary.json` es la fuente de
verdad del mapeo clase→índice y del `image_size` con que se entrenó el checkpoint.

## Modelos baseline

Se adoptan tres arquitecturas ligeras que cubren el eje precisión↔eficiencia y convierten a
TensorFlow Lite para despliegue móvil offline:

- `efficientnet_b0`
- `shufflenet_v2_x1_0`
- `efficientnet_lite0`

## Ejecución

```bash
# Local
make train-baselines MODELS=all EPOCHS=30

# GPU en Modal (cap 1500 sobre 9 clases; regenera splits si cambió el perfil)
make modal-train-baselines MODELS=all EPOCHS=30 MAX_PER_CLASS=1500 REGEN_SPLITS=1
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/es/pipeline-baselines/entrenamiento.md
git commit -m "docs(baselines): pagina de entrenamiento (config canonica 9 clases)"
```

### Task C3: `pipeline-baselines/evaluacion.md`

**Files:**
- Modify: `docs/es/pipeline-baselines/evaluacion.md`

- [ ] **Step 1: Reemplazar el contenido completo**

```md
# Evaluación

Cada baseline se evalúa sobre `test.csv` (N = 1 503, 9 clases) con transformaciones
deterministas. La métrica primaria es **macro-F1** (pondera por igual todas las clases, incluidas
las minoritarias); accuracy es secundaria por el desbalance.

## Resultados (test, 9 clases)

| Modelo | Accuracy | macro-F1 |
|---|---:|---:|
| `efficientnet_b0` | 0.9521 | **0.9146** |
| `shufflenet_v2_x1_0` | 0.9508 | 0.9030 |
| `efficientnet_lite0` | 0.9474 | 0.8951 |

`efficientnet_b0` lidera el macro-F1 y es el umbral de referencia actual.

## F1 por clase

| Clase | b0 | shufflenet | lite0 |
|---|---:|---:|---:|
| common_rust | 0.99 | 0.99 | 0.98 |
| fall_armyworm | 0.96 | 0.96 | 0.95 |
| gray_leaf_spot | 0.95 | 0.95 | 0.95 |
| healthy | 0.96 | 0.96 | 0.97 |
| lethal_necrosis | 0.99 | 0.99 | 1.00 |
| nitrogen_deficiency | 0.87 | 0.88 | 0.85 |
| northern_corn_leaf_blight | 0.95 | 0.94 | 0.94 |
| phosphorus_deficiency | 0.95 | 0.92 | 0.92 |
| **potassium_deficiency** | **0.62** | **0.54** | **0.49** |

## Hallazgos

- **`potassium_deficiency` es el cuello de botella universal** (266 imágenes totales, 40 en test):
  recall muy bajo en todos los modelos y responsable de la mayor caída del macro-F1.
- **Clúster de deficiencias N/P/K:** el potasio se confunde sobre todo con nitrógeno (en
  `efficientnet_b0`: de 40 imágenes de potasio, 21 correctas, 13→nitrógeno, 5→fósforo). Las tres
  deficiencias comparten síntomas de clorosis y son difíciles de separar con pocos ejemplos.
- **Clúster de lesiones foliares:** `gray_leaf_spot` ↔ `northern_corn_leaf_blight` se confunden
  mutuamente (lesiones visualmente parecidas).
- **Clases fáciles:** `common_rust`, `lethal_necrosis` y `healthy` (f1 ≥ 0.96 en todos los modelos).

La palanca de mayor impacto para el modelo final es aumentar/rebalancear las deficiencias
(especialmente potasio) antes de subir capacidad de modelo.
```

- [ ] **Step 2: Commit**

```bash
git add docs/es/pipeline-baselines/evaluacion.md
git commit -m "docs(baselines): pagina de evaluacion (resultados 9 clases + hallazgos)"
```

### Task C4: `pipeline-baselines/interpretabilidad.md`

**Files:**
- Modify: `docs/es/pipeline-baselines/interpretabilidad.md`

- [ ] **Step 1: Reemplazar el contenido completo**

```md
# Interpretabilidad

La explicabilidad es **post-hoc** y no está acoplada al entrenamiento: se corre sobre checkpoints
ya entrenados con `best.pth`. Combina LIME (regiones de superpíxeles que sostienen el diagnóstico)
y Grad-CAM (mapa de activación de la clase predicha).

## Scripts

| Script | Salida |
|---|---|
| `explain_lime.py` (`make explain-lime`) | Reporte visual LIME + Grad-CAM por imagen |
| `explain_report.py` (`make explain-report`) | Fidelidad agregada y dispersión por clase |
| `explain_report.py --errors-only` (`make explain-errors`) | LIME dirigido a errores (label ≠ pred) |

## Consistencia de etiquetas

El mapeo clase→índice y el `image_size` de cada reporte se leen del `summary.json` del run
(fuente de verdad del head entrenado), no se reconstruyen desde el YAML. Esto garantiza que la
etiqueta mostrada en el panel coincida con la predicción real del modelo.

## Qué observar

- En las clases fáciles (roya, necrosis letal) los mapas se concentran en el tejido de la hoja.
- En las deficiencias N/P/K y en imágenes de campo con fondo cargado, la atribución puede
  dispersarse hacia el fondo — señal de contexto espurio a vigilar, coherente con la confusión
  entre deficiencias observada en la evaluación.
```

- [ ] **Step 2: Commit**

```bash
git add docs/es/pipeline-baselines/interpretabilidad.md
git commit -m "docs(baselines): pagina de interpretabilidad (LIME + Grad-CAM post-hoc)"
```

### Task C5: `pipeline-baselines/experimentos.md` (ÚNICO lugar con la exploración)

**Files:**
- Modify: `docs/es/pipeline-baselines/experimentos.md`

**Interfaces:**
- Produces: la única página que menciona 4 clases, cap 500 / sin cap, y las 8 arquitecturas.

- [ ] **Step 1: Reemplazar el contenido completo**

```md
# Experimentos

Antes de fijar el baseline canónico de 9 clases, se realizó una **exploración de arquitecturas**
sobre un subconjunto reducido de **4 clases** (`healthy`, `common_rust`, `fall_armyworm`,
`nitrogen_deficiency`), en dos regímenes de datos:

- **Cap de 500 imágenes por clase** — comparación rápida y barata + sonda de eficiencia muestral.
- **Sin límite por clase** — mismas 4 clases con todas las imágenes disponibles.

El objetivo fue comparar el comportamiento de cada arquitectura (convergencia, colapso de clases,
sensibilidad al volumen de datos) a bajo costo, no medir el rendimiento final.

## Arquitecturas exploradas

Se probaron hasta **8 arquitecturas ligeras**:

- `efficientnet_b0`
- `efficientnet_b4`
- `efficientnet_lite0`
- `fastvit_t8`
- `ghostnetv2_100`
- `mobilenet_v3_large`
- `mobilenet_v3_small`
- `shufflenet_v2_x1_0`

De esta exploración se adoptaron como baselines definitivos `efficientnet_b0`,
`shufflenet_v2_x1_0` y `efficientnet_lite0` (ver [Baselines](../baselines/)).

## Resultados completos

Los outputs completos de estas corridas exploratorias (métricas, matrices de confusión, reportes
LIME) se archivan por separado:

<!-- TODO(davidderas50): subir el ZIP y pegar el enlace de Google Drive -->
📦 **Resultados de experimentos (4 clases):** [ZIP en Google Drive](URL_PENDIENTE)
```

- [ ] **Step 2: Commit**

```bash
git add docs/es/pipeline-baselines/experimentos.md
git commit -m "docs(baselines): pagina de experimentos (exploracion 4 clases, 8 arquitecturas)"
```

---

## Fase D — README, navegación y preprocesado compartido

### Task D1: Actualizar `README.md`

**Files:**
- Modify: `README.md` (líneas ~58, ~105-107, ~155-156, ~181, ~203)

- [ ] **Step 1: Línea ~58 — arquitectura base**

Reemplazar:
```md
| Arquitectura base | En evaluación entre 6 baselines: EfficientNet-B0/Lite0, MobileNetV3-Large, FastViT-T8, GhostNetV2-100, ShuffleNetV2-x1.0 |
```
por:
```md
| Arquitectura base | 3 baselines sobre 9 clases: EfficientNet-B0 (líder, macro-F1 0.915), ShuffleNetV2-x1.0, EfficientNet-Lite0 |
```

- [ ] **Step 2: Líneas ~105-107 — descripción del baseline**

Reemplazar la descripción que menciona "6 baselines … 4 clases y hasta 500 imágenes por clase" por una que diga: baselines entrenados sobre el perfil `baseline` (9 clases, cap 1 500/clase) con 3 arquitecturas (`efficientnet_b0`, `shufflenet_v2_x1_0`, `efficientnet_lite0`).

- [ ] **Step 3: Líneas ~155-156 — quitar bloque vast**

Eliminar:
```md
Para GPU alquilada por SSH en [vast.ai](https://vast.ai) en vez de Modal, ver
[docs/es/deployment/vast-ai.md](docs/es/deployment/vast-ai.md).
```

- [ ] **Step 4: Línea ~181 — árbol del proyecto**

Eliminar la línea `│   └── vastai/           # Orquestación de GPU remota en vast.ai` y ajustar el conector del árbol para que `modal/` sea la última entrada de `scripts/`. Cambiar también `└── models/  # Registro de modelos + 6 baselines` → `+ 8 arquitecturas registradas (3 baselines)` y `Dockerfile ...` (eliminar esa línea del árbol).

- [ ] **Step 5: Línea ~203 — estado del proyecto**

Reemplazar:
```md
- [x] Entrenamiento de 6 baselines (EfficientNet-B0/Lite0, MobileNetV3-Large, FastViT-T8, GhostNetV2-100, ShuffleNetV2-x1.0) + soporte GPU remota (vast.ai, Modal)
```
por:
```md
- [x] Baselines sobre 9 clases (EfficientNet-B0, ShuffleNetV2-x1.0, EfficientNet-Lite0; cap 1 500/clase) + entrenamiento en GPU de Modal
```

- [ ] **Step 6: Verificar README limpio**

Run: `grep -niE "vast|6 baselines|500 imágenes|Dockerfile" README.md`
Expected: sin resultados.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs(readme): 3 baselines sobre 9 clases; quita vast.ai y Dockerfile"
```

### Task D2: Enlazar `baselines/index.md` en la nav + ajustar preprocesado compartido

**Files:**
- Modify: `docs/.vitepress/config.mts` (grupo Baselines en sidebar y nav)
- Modify: `docs/es/preprocesado/index.md`

- [ ] **Step 1: Añadir "Modelos" como primer ítem del grupo Baselines (sidebar y nav)**

En `config.mts`, dentro de los DOS bloques del grupo `Baselines` (uno en `esDatasetSidebar`, otro en `themeConfig.nav`), añadir como primer `item`:

```ts
          { text: "Modelos", link: "/es/baselines/" },
```

quedando el grupo: Modelos → Preprocesado → Entrenamiento → Evaluación → Interpretabilidad → Experimentos.

- [ ] **Step 2: Corregir "8 clases" → "9 clases" en el preprocesado compartido**

En `docs/es/preprocesado/index.md:40`, cambiar `clasificación de hojas con 8 clases` por `clasificación de hojas con 9 clases`.

- [ ] **Step 3: Añadir nota del baseline al final de "## División"**

Tras la línea de CSV inmutables (`:18`), añadir:

```md
- **Baseline vs. principal:** el pipeline principal usa `seed_42/` (9 clases sin cap); el baseline usa `seed_42_baseline/` (9 clases con cap de 1 500/clase). Misma estratificación y seed.
```

- [ ] **Step 4: Commit**

```bash
git add docs/.vitepress/config.mts docs/es/preprocesado/index.md
git commit -m "docs: enlaza overview de modelos y aclara split baseline vs principal"
```

---

## Fase E — Verificación final

### Task E1: Build de docs y barridos de consistencia

- [ ] **Step 1: Build de VitePress (sin enlaces rotos)**

Run: `npm run docs:build`
Expected: build OK, sin "dead link" a `vast-ai` ni a páginas borradas.

- [ ] **Step 2: Barrido — sin vast en fuentes vivas**

Run: `grep -rniE "vast" . --include="*.md" --include="*.mts" --include="*.py" --include="*.example" 2>/dev/null | grep -v venv | grep -v "/dist/" | grep -v ".superpowers/" | grep -v "docs/superpowers/plans/"`
Expected: sin resultados.

- [ ] **Step 3: Barrido — sin datos de experimento fuera de experimentos.md**

Run: `grep -rniE "4 clases|cuatro clases|500 imágenes|mobilenet_v3|fastvit|ghostnet|efficientnet_b4|mobilenet_v3_small" docs/es/baselines docs/es/pipeline-baselines/entrenamiento.md docs/es/pipeline-baselines/evaluacion.md docs/es/pipeline-baselines/interpretabilidad.md docs/es/pipeline-baselines/preprocesado.md docs/es/preprocesado 2>/dev/null`
Expected: sin resultados (la exploración vive solo en `experimentos.md`).

- [ ] **Step 4: Confirmar los 3 baselines en la doc canónica**

Run: `grep -rlniE "shufflenet_v2_x1_0|efficientnet_b0|efficientnet_lite0" docs/es/baselines/index.md docs/es/pipeline-baselines/entrenamiento.md docs/es/pipeline-baselines/evaluacion.md`
Expected: los tres archivos aparecen.

- [ ] **Step 5: Commit final (si el build generó lockfiles u otros)**

```bash
git add -A
git commit -m "chore: verify docs build tras baselines 9 clases y remocion vast.ai" || echo "nada que commitear"
```

---

## Self-Review (cobertura del spec)

- **(1) Quitar vast.ai** → Fase A (A1 borra código/Dockerfile/doc; A2 limpia nav/prosa/env/agent). ✅
- **(2) Solo 3 baselines** → B1 (swap mobilenet→shufflenet, tabla comparativa), D1 (README). ✅
- **(3) Mencionar experimentos 4 clases + 8 modelos + link ZIP** → C5, confinado a `experimentos.md`. ✅
- **(4) Canónico = corrida 9 clases cap 1500, sin referenciar experimentos** → Global Constraints + B1/C1-C4/D2, con barrido E1-Step3. ✅
- **(5) Secciones actualizadas** → preprocesado compartido (D2), Baselines: preprocesado (C1), entrenamiento (C2), evaluación (C3), interpretabilidad (C4), experimentos (C5). ✅

**Placeholder intencional:** el enlace `URL_PENDIENTE` en `experimentos.md` es el ZIP futuro que el usuario pidió dejar; queda marcado con un `TODO` explícito para completar al subirlo.
