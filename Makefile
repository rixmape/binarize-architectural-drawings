PYENV_ROOT ?= $(HOME)/.pyenv
PYENV      ?= $(PYENV_ROOT)/bin/pyenv
PY_VERSION := $(shell cat .python-version)
PY_BIN     := $(PYENV_ROOT)/versions/$(PY_VERSION)/bin/python
PYTHON     ?= venv/bin/python
PIP        ?= venv/bin/pip
SCRIPT     := binarize.py
LINT_MAX   := 120

.PHONY: help pyenv venv install lint test clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

pyenv: ## Install the required Python version via pyenv
	$(PYENV) install -s $(PY_VERSION)

venv: pyenv ## Create virtual environment and install dependencies
	$(PY_BIN) -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: ## Install dependencies
	$(PIP) install -r requirements.txt

lint: ## Run PEP 8 check
	$(PIP) install -q pycodestyle
	pycodestyle --max-line-length=$(LINT_MAX) $(SCRIPT)

test:  ## Smoke-test all 8 model architectures
	$(PYTHON) -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import binarize; [binarize.build_model(i) for i in range(1, 9)]; print('All 8 models OK')"

clean: ## Remove generated artifacts
	rm -rf runs/ results/ __pycache__/ .pytest_cache/
	find . -name '*.pyc' -delete
