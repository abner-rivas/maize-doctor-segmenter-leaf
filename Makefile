# Usa el intérprete del entorno activo en Linux, macOS y Windows.
# Todas las variables son sobreescribibles desde la CLI.
PYTHON ?= python
PIP ?= $(PYTHON) -m pip
RUFF ?= $(PYTHON) -m ruff
PYRIGHT ?= $(PYTHON) -m pyright
MODAL ?= $(PYTHON) -m modal

MODELS ?= efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0
EPOCHS ?= 30
NO_CAP ?=
MAX_PER_CLASS ?=
REGEN_SPLITS ?=
BATCH_SIZE ?=
IMAGE_SIZE ?=
LEARNING_RATE ?=
WEIGHT_DECAY ?=
NUM_WORKERS ?=
NO_PRETRAINED ?=
LIME ?=
NUM_SAMPLES ?=
CONFIRM_TRAINING ?=
CONFIRM_SEGMENTATION_TRAINING ?=
CONFIRM_SEGMENTATION_SMOKE_TRAINING ?=
CONFIRM_PILOT_EVALUATION ?=
CONFIRM_CLEAN_OUTPUTS ?=

CLOUD_TRAINING_DIR ?= cloud_training
LEAF_SEGMENTATION_DATASET ?= data/leaf_detection/detector_dataset
LEAF_SEGMENTATION_OUTPUT ?= outputs/leaf_detection
LEAF_SEGMENTATION_PACKAGE_DIR ?= $(LEAF_SEGMENTATION_OUTPUT)/packages
SEGMENTATION_MODEL ?= yolo26n-seg.pt
SEGMENTATION_DEVICE ?= 0
PACKAGE ?=

.PHONY: help \
	leaf-segmentation-status leaf-segmentation-verify-locks \
	leaf-segmentation-verify-splits leaf-segmentation-preflight \
	leaf-segmentation-cloud-package leaf-segmentation-cloud-package-verify \
	leaf-segmentation-cloud-package-list leaf-segmentation-cloud-clean-temp \
	leaf-segmentation-cloud-bootstrap leaf-segmentation-cloud-preflight \
	leaf-segmentation-cloud-smoke leaf-segmentation-cloud-train \
	leaf-segmentation-cloud-resume leaf-segmentation-cloud-validate \
	leaf-segmentation-cloud-test leaf-segmentation-cloud-results \
	leaf-segmentation-cloud-checksums leaf-segmentation-pilot-evaluate \
	leaf-segmentation-cloud-prepare leaf-segmentation-cloud-check \
	compile-pdf install download-dataset splits splits-baseline train \
	train-baselines explain-lime explain-report explain-errors test-loader \
	smoke-loader audit-dataset validate-splits training-preflight \
	training-package-manifest summary docs-eda lint lint-fix fmt check \
	clean-outputs modal-seed modal-train-baselines modal-clean-outputs \
	modal-explain-lime modal-explain-report modal-explain-errors modal-pull

LEAF_SEGMENTATION_MAKE_HELPER = $(PYTHON) scripts/package/leaf_segmentation_make.py \
	--dataset "$(LEAF_SEGMENTATION_DATASET)" \
	--output "$(LEAF_SEGMENTATION_OUTPUT)" \
	--cloud-dir "$(CLOUD_TRAINING_DIR)" \
	--package-dir "$(LEAF_SEGMENTATION_PACKAGE_DIR)" \
	--model "$(SEGMENTATION_MODEL)" \
	--device "$(SEGMENTATION_DEVICE)" \
	$(if $(PACKAGE),--package "$(PACKAGE)",)

define REQUIRE_TRAINING_CONFIRMATION
$(if $(filter 1,$(CONFIRM_TRAINING)),,$(error Entrenamiento no iniciado. Use CONFIRM_TRAINING=1 para confirmar explícitamente.))
endef

define REQUIRE_SEGMENTATION_TRAINING_CONFIRMATION
$(if $(and $(filter 1,$(strip $(CONFIRM_SEGMENTATION_TRAINING))),$(filter 1,$(words $(strip $(CONFIRM_SEGMENTATION_TRAINING))))),,$(error ERROR: entrenamiento completo no autorizado. Ejecute: CONFIRM_SEGMENTATION_TRAINING=1 make leaf-segmentation-cloud-train))
endef

define REQUIRE_SEGMENTATION_SMOKE_CONFIRMATION
$(if $(and $(filter 1,$(strip $(CONFIRM_SEGMENTATION_SMOKE_TRAINING))),$(filter 1,$(words $(strip $(CONFIRM_SEGMENTATION_SMOKE_TRAINING))))),,$(error ERROR: entrenamiento smoke no autorizado. Ejecute: CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 make leaf-segmentation-cloud-smoke))
endef

define REQUIRE_PILOT_EVALUATION_CONFIRMATION
$(if $(and $(filter 1,$(strip $(CONFIRM_PILOT_EVALUATION))),$(filter 1,$(words $(strip $(CONFIRM_PILOT_EVALUATION))))),,$(error ERROR: evaluación del piloto no autorizada. Ejecute: CONFIRM_PILOT_EVALUATION=1 make leaf-segmentation-pilot-evaluate))
endef

help:
	@printf '%s\n' \
		'DoctorMaiz — interfaz de segmentación' \
		'' \
		'LOCAL / SEGURO:' \
		'  leaf-segmentation-status                 Estado sin modificar archivos' \
		'  leaf-segmentation-verify-locks           Locks y fingerprints' \
		'  leaf-segmentation-verify-splits          Dataset/splits sin reconstruir' \
		'  leaf-segmentation-preflight              Auditoría local sin instalar' \
		'  leaf-segmentation-cloud-package          Construir paquete determinista' \
		'  leaf-segmentation-cloud-package-verify   Verificar/extractar PACKAGE=<ruta>' \
		'  leaf-segmentation-cloud-package-list     Listar paquetes y tamaños' \
		'  leaf-segmentation-cloud-clean-temp       Borrar sólo temporales del packager' \
		'  leaf-segmentation-cloud-prepare          Locks + splits + package + verify' \
		'  leaf-segmentation-cloud-check            Status + locks + splits' \
		'' \
		'CLOUD / SIN ENTRENAR:' \
		'  leaf-segmentation-cloud-bootstrap        Instalar en entorno cloud aislado' \
		'  leaf-segmentation-cloud-preflight        GPU, modelo, pesos y forward' \
		'  leaf-segmentation-cloud-validate         best.pt sobre val' \
		'  leaf-segmentation-cloud-test             best.pt sobre test interno' \
		'  leaf-segmentation-cloud-results          Mostrar resultados sin cambiarlos' \
		'  leaf-segmentation-cloud-checksums        Hashes de resultados' \
		'' \
		'ENTRENAMIENTO / CONFIRMACIÓN OBLIGATORIA:' \
		'  leaf-segmentation-cloud-smoke   CONFIRM_SEGMENTATION_SMOKE_TRAINING=1' \
		'  leaf-segmentation-cloud-train   CONFIRM_SEGMENTATION_TRAINING=1' \
		'  leaf-segmentation-cloud-resume  CONFIRM_SEGMENTATION_TRAINING=1' \
		'  leaf-segmentation-pilot-evaluate CONFIRM_PILOT_EVALUATION=1'

install:
	$(PIP) install -e ".[dev,analysis,xai,cloud]"

download-dataset:
	$(PYTHON) scripts/dataset/download_dataset.py

splits:
	$(PYTHON) scripts/pipeline/create_splits.py

splits-baseline:
	$(PYTHON) scripts/pipeline/create_splits.py --baseline $(if $(NO_CAP),--no-cap,) $(if $(MAX_PER_CLASS),--max-per-class $(MAX_PER_CLASS),)

train:
	$(REQUIRE_TRAINING_CONFIRMATION)
	$(PYTHON) scripts/pipeline/train.py

train-baselines:
	$(REQUIRE_TRAINING_CONFIRMATION)
	$(PYTHON) scripts/pipeline/train_baselines.py --models $(MODELS) --baseline \
		$(if $(NO_CAP),--no-cap,) $(if $(MAX_PER_CLASS),--max-per-class $(MAX_PER_CLASS),) \
		$(if $(REGEN_SPLITS),--regenerate-splits,) \
		--epochs $(EPOCHS) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(IMAGE_SIZE),--image-size $(IMAGE_SIZE),) \
		$(if $(LEARNING_RATE),--learning-rate $(LEARNING_RATE),) \
		$(if $(WEIGHT_DECAY),--weight-decay $(WEIGHT_DECAY),) \
		$(if $(NUM_WORKERS),--num-workers $(NUM_WORKERS),) \
		$(if $(NO_PRETRAINED),--no-pretrained,) \
		$(if $(LIME),--lime,)

# Protegido: outputs/ contiene evidencia de auditoría, paquetes cloud (~2 GiB)
# y, en el futuro, checkpoints entrenados que no deben borrarse por accidente.
clean-outputs:
	$(if $(filter 1,$(strip $(CONFIRM_CLEAN_OUTPUTS))),,$(error ERROR: borrado de outputs/ no autorizado. Ejecute: CONFIRM_CLEAN_OUTPUTS=1 make clean-outputs))
	rm -rf outputs/

modal-seed:
	$(MODAL) run scripts/modal/train.py::seed_dataset

modal-train-baselines:
	$(REQUIRE_TRAINING_CONFIRMATION)
	$(MODAL) run scripts/modal/train.py --models "$(MODELS)" --epochs "$(EPOCHS)" \
		$(if $(NO_CAP),--no-cap,) $(if $(MAX_PER_CLASS),--max-per-class "$(MAX_PER_CLASS)",) \
		$(if $(REGEN_SPLITS),--regenerate-splits,) \
		$(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) \
		$(if $(IMAGE_SIZE),--image-size "$(IMAGE_SIZE)",) \
		$(if $(LEARNING_RATE),--learning-rate "$(LEARNING_RATE)",) \
		$(if $(WEIGHT_DECAY),--weight-decay "$(WEIGHT_DECAY)",) \
		$(if $(NUM_WORKERS),--num-workers "$(NUM_WORKERS)",) \
		$(if $(NO_PRETRAINED),--no-pretrained,) \
		$(if $(LIME),--lime,)

modal-clean-outputs:
	$(MODAL) run scripts/modal/train.py::clean_outputs

modal-explain-lime:
	$(MODAL) run scripts/modal/explain.py::explain_lime --models "$(MODELS)" \
		$(if $(RUN),--run $(RUN),) $(if $(IMAGE),--image $(IMAGE),) $(if $(OUTPUT),--output $(OUTPUT),)

modal-explain-report:
	$(MODAL) run scripts/modal/explain.py::explain_report --models "$(MODELS)" \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

modal-explain-errors:
	$(MODAL) run scripts/modal/explain.py::explain_errors --models "$(MODELS)" \
		$(if $(RUN),--run $(RUN),) $(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

modal-pull:
	$(MODAL) volume get --force corn-outputs / ./outputs-remote

explain-lime:
	$(PYTHON) scripts/pipeline/explain_lime.py --models $(MODELS) $(if $(IMAGE),--image $(IMAGE),) $(if $(OUTPUT),--output $(OUTPUT),)

explain-report:
	$(PYTHON) scripts/pipeline/explain_report.py --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

explain-errors:
	$(PYTHON) scripts/pipeline/explain_report.py --models $(MODELS) --errors-only \
		$(if $(RUN),--run $(RUN),) $(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

test-loader:
	$(PYTHON) scripts/checks/smoke_loader.py

smoke-loader: test-loader

audit-dataset:
	$(PYTHON) scripts/checks/audit_dataset_classes.py --fail-on-mismatch

validate-splits:
	$(PYTHON) scripts/checks/validate_splits.py --fail-on-error

training-preflight:
	$(PYTHON) scripts/checks/training_preflight.py --device cpu --check-dataset --output outputs/preflight

leaf-segmentation-preflight:
	$(PYTHON) scripts/pipeline/leaf_segmentation_preflight.py \
		--dataset-root "$(LEAF_SEGMENTATION_DATASET)" \
		--output-root "$(LEAF_SEGMENTATION_OUTPUT)/training_preflight"

leaf-segmentation-status:
	$(LEAF_SEGMENTATION_MAKE_HELPER) status

leaf-segmentation-verify-locks:
	$(LEAF_SEGMENTATION_MAKE_HELPER) verify-locks

leaf-segmentation-verify-splits:
	$(LEAF_SEGMENTATION_MAKE_HELPER) verify-splits

leaf-segmentation-cloud-package: leaf-segmentation-verify-locks
	$(PYTHON) scripts/package/build_leaf_segmentation_cloud_package.py \
		--dataset-root "$(LEAF_SEGMENTATION_DATASET)" \
		--output-dir "$(LEAF_SEGMENTATION_PACKAGE_DIR)"

leaf-segmentation-cloud-package-verify:
	$(LEAF_SEGMENTATION_MAKE_HELPER) package-verify

leaf-segmentation-cloud-package-list:
	$(LEAF_SEGMENTATION_MAKE_HELPER) package-list

leaf-segmentation-cloud-clean-temp:
	$(LEAF_SEGMENTATION_MAKE_HELPER) clean-temp

leaf-segmentation-cloud-bootstrap:
	PYTHON="$(PYTHON)" CLOUD_TRAINING_DIR="$(CLOUD_TRAINING_DIR)" \
		LEAF_SEGMENTATION_DATASET="$(LEAF_SEGMENTATION_DATASET)" \
		LEAF_SEGMENTATION_OUTPUT="$(LEAF_SEGMENTATION_OUTPUT)" \
		SEGMENTATION_MODEL="$(SEGMENTATION_MODEL)" \
		SEGMENTATION_DEVICE="$(SEGMENTATION_DEVICE)" \
		bash "$(CLOUD_TRAINING_DIR)/bootstrap_cloud.sh"

leaf-segmentation-cloud-preflight:
	PYTHON="$(PYTHON)" CLOUD_TRAINING_DIR="$(CLOUD_TRAINING_DIR)" \
		LEAF_SEGMENTATION_DATASET="$(LEAF_SEGMENTATION_DATASET)" \
		LEAF_SEGMENTATION_OUTPUT="$(LEAF_SEGMENTATION_OUTPUT)" \
		SEGMENTATION_MODEL="$(SEGMENTATION_MODEL)" \
		SEGMENTATION_DEVICE="$(SEGMENTATION_DEVICE)" \
		bash "$(CLOUD_TRAINING_DIR)/preflight_cloud.sh"

leaf-segmentation-cloud-smoke:
	$(REQUIRE_SEGMENTATION_SMOKE_CONFIRMATION)
	CONFIRM_SEGMENTATION_SMOKE_TRAINING=$(CONFIRM_SEGMENTATION_SMOKE_TRAINING) \
		PYTHON="$(PYTHON)" CLOUD_TRAINING_DIR="$(CLOUD_TRAINING_DIR)" \
		LEAF_SEGMENTATION_DATASET="$(LEAF_SEGMENTATION_DATASET)" \
		LEAF_SEGMENTATION_OUTPUT="$(LEAF_SEGMENTATION_OUTPUT)" \
		SEGMENTATION_MODEL="$(SEGMENTATION_MODEL)" \
		SEGMENTATION_DEVICE="$(SEGMENTATION_DEVICE)" \
		bash "$(CLOUD_TRAINING_DIR)/smoke_train.sh"

leaf-segmentation-cloud-train:
	$(REQUIRE_SEGMENTATION_TRAINING_CONFIRMATION)
	CONFIRM_SEGMENTATION_TRAINING=$(CONFIRM_SEGMENTATION_TRAINING) \
		PYTHON="$(PYTHON)" CLOUD_TRAINING_DIR="$(CLOUD_TRAINING_DIR)" \
		LEAF_SEGMENTATION_DATASET="$(LEAF_SEGMENTATION_DATASET)" \
		LEAF_SEGMENTATION_OUTPUT="$(LEAF_SEGMENTATION_OUTPUT)" \
		SEGMENTATION_MODEL="$(SEGMENTATION_MODEL)" \
		SEGMENTATION_DEVICE="$(SEGMENTATION_DEVICE)" \
		bash "$(CLOUD_TRAINING_DIR)/train.sh"

leaf-segmentation-cloud-resume:
	$(REQUIRE_SEGMENTATION_TRAINING_CONFIRMATION)
	CONFIRM_SEGMENTATION_TRAINING=$(CONFIRM_SEGMENTATION_TRAINING) \
		PYTHON="$(PYTHON)" CLOUD_TRAINING_DIR="$(CLOUD_TRAINING_DIR)" \
		LEAF_SEGMENTATION_DATASET="$(LEAF_SEGMENTATION_DATASET)" \
		LEAF_SEGMENTATION_OUTPUT="$(LEAF_SEGMENTATION_OUTPUT)" \
		SEGMENTATION_MODEL="$(SEGMENTATION_MODEL)" \
		SEGMENTATION_DEVICE="$(SEGMENTATION_DEVICE)" \
		bash "$(CLOUD_TRAINING_DIR)/resume_train.sh"

leaf-segmentation-cloud-validate:
	PYTHON="$(PYTHON)" CLOUD_TRAINING_DIR="$(CLOUD_TRAINING_DIR)" \
		LEAF_SEGMENTATION_DATASET="$(LEAF_SEGMENTATION_DATASET)" \
		LEAF_SEGMENTATION_OUTPUT="$(LEAF_SEGMENTATION_OUTPUT)" \
		SEGMENTATION_MODEL="$(SEGMENTATION_MODEL)" \
		SEGMENTATION_DEVICE="$(SEGMENTATION_DEVICE)" \
		bash "$(CLOUD_TRAINING_DIR)/validate.sh"

leaf-segmentation-cloud-test:
	PYTHON="$(PYTHON)" CLOUD_TRAINING_DIR="$(CLOUD_TRAINING_DIR)" \
		LEAF_SEGMENTATION_DATASET="$(LEAF_SEGMENTATION_DATASET)" \
		LEAF_SEGMENTATION_OUTPUT="$(LEAF_SEGMENTATION_OUTPUT)" \
		SEGMENTATION_MODEL="$(SEGMENTATION_MODEL)" \
		SEGMENTATION_DEVICE="$(SEGMENTATION_DEVICE)" \
		bash "$(CLOUD_TRAINING_DIR)/evaluate_test.sh"

leaf-segmentation-cloud-results:
	$(LEAF_SEGMENTATION_MAKE_HELPER) results

leaf-segmentation-cloud-checksums:
	$(LEAF_SEGMENTATION_MAKE_HELPER) checksums

leaf-segmentation-pilot-evaluate:
	$(REQUIRE_PILOT_EVALUATION_CONFIRMATION)
	$(LEAF_SEGMENTATION_MAKE_HELPER) pilot-gate
	LEAF_SEGMENTATION_OUTPUT="$(LEAF_SEGMENTATION_OUTPUT)" \
		SEGMENTATION_DEVICE="$(SEGMENTATION_DEVICE)" \
		$(PYTHON) scripts/pipeline/leaf_segmentation_pilot_evaluate.py

leaf-segmentation-cloud-prepare: leaf-segmentation-verify-locks \
	leaf-segmentation-verify-splits leaf-segmentation-cloud-package \
	leaf-segmentation-cloud-package-verify

leaf-segmentation-cloud-check: leaf-segmentation-status \
	leaf-segmentation-verify-locks leaf-segmentation-verify-splits

training-package-manifest:
	$(PYTHON) scripts/checks/build_training_package_manifest.py --output outputs/training_package_manifest.json

summary:
	$(PYTHON) src/analysis/dataset_summary.py

# Requiere shell POSIX (Powershell/Git Bash/WSL en Windows) y que existan outputs/eda/eda_*.png
docs-eda:
	cp outputs/eda/eda_*.png public/eda/

lint:
	$(RUFF) check src/ scripts/

lint-fix:
	$(RUFF) check --fix src/ scripts/

fmt:
	$(RUFF) format src/ scripts/

check:
	$(PYRIGHT) src/ scripts/

compile-pdf:
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex
	cd reports/firts-phase && pdflatex -interaction=nonstopmode documentation_first_phase.tex
