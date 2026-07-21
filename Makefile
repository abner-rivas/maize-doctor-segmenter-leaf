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

.PHONY: compile-pdf install download-dataset splits splits-baseline train train-baselines explain-lime explain-report explain-errors test-loader smoke-loader audit-dataset validate-splits training-preflight training-package-manifest summary docs-eda lint lint-fix fmt check clean-outputs modal-seed modal-train-baselines modal-clean-outputs modal-explain-lime modal-explain-report modal-explain-errors modal-pull

define REQUIRE_TRAINING_CONFIRMATION
$(if $(filter 1,$(CONFIRM_TRAINING)),,$(error Entrenamiento no iniciado. Use CONFIRM_TRAINING=1 para confirmar explícitamente.))
endef

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

clean-outputs:
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
