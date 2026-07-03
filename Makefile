ifeq ($(OS),Windows_NT)
    PYTHON 	:= venv\Scripts\python
    PIP    	:= venv\Scripts\pip
    RUFF   	:= venv\Scripts\ruff
	PYRIGHT := venv\Scripts\pyright
else
    PYTHON 	:= venv/bin/python
    PIP    	:= venv/bin/pip
    RUFF   	:= venv/bin/ruff
	PYRIGHT := venv/bin/pyright
endif

MODELS ?= all

.PHONY: install download-dataset splits splits-baseline train train-baselines train-baselines-full explain-lime test-loader summary docs-eda lint fmt

install:
	$(PIP) install -e ".[dev,analysis,xai]"

download-dataset:
	$(PYTHON) scripts/dataset/download_dataset.py

splits:
	$(PYTHON) scripts/pipeline/create_splits.py

splits-baseline:
	$(PYTHON) scripts/pipeline/create_splits.py --baseline

train:
	$(PYTHON) scripts/pipeline/train.py

train-baselines:
	$(PYTHON) scripts/pipeline/train_baselines.py --models $(MODELS) --baseline --lime

train-baselines-full:
	$(PYTHON) scripts/pipeline/train_baselines.py --models $(MODELS) --lime

# Reporte visual LIME de 3 paneles. Por defecto, muestreo balanceado del test set para
# cada modelo de MODELS (config/dataset.yaml -> lime:). Con IMAGE=ruta/a/imagen.jpg genera
# un único reporte puntual por modelo. OUTPUT solo es válido junto con IMAGE y un MODELS de uno solo.
explain-lime:
	$(PYTHON) scripts/pipeline/explain_lime.py --models $(MODELS) $(if $(IMAGE),--image $(IMAGE),) $(if $(OUTPUT),--output $(OUTPUT),)

summary:
	$(PYTHON) src/analysis/dataset_summary.py

docs-eda:
	cp tmp/eda_*.png public/eda/

lint:
	$(RUFF) check src/ scripts/

lint-fix:
	$(RUFF) check --fix src/ scripts/

fmt:
	$(RUFF) format src/ scripts/

check:
	$(PYRIGHT) src/ scripts/
