SHELL := /bin/bash

CONFIG ?= configs/smoke.yaml
VENV ?= .venv
PY ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

.PHONY: setup data train eval report all clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@./scripts/bootstrap_venv.sh "$(VENV)"
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt

setup-ml: setup
	@$(PIP) install -r requirements-ml.txt

data:
	@PYTHONPATH=src $(PY) scripts/data.py --config "$(CONFIG)"

train:
	@PYTHONPATH=src $(PY) scripts/train.py --config "$(CONFIG)"

eval:
	@PYTHONPATH=src $(PY) scripts/eval.py --config "$(CONFIG)"

report:
	@PYTHONPATH=src $(PY) scripts/report.py --config "$(CONFIG)"

all: setup data train eval report

clean:
	@python3 -c 'import shutil; from pathlib import Path; p=Path("artifacts"); shutil.rmtree(p, ignore_errors=True); p.mkdir(parents=True, exist_ok=True)'
