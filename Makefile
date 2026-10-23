# IndicMix AI - Project Automation Makefile

.PHONY: help install datasets train export quantize test test-node benchmark serve

PYTHON ?= python
PIP ?= pip

help:
	@echo "IndicMix AI Automation Commands:"
	@echo "  make install     - Install Python package and development dependencies"
	@echo "  make datasets    - Generate and split leak-free synthetic datasets"
	@echo "  make train       - Train classification, PII, and normalizer models"
	@echo "  make export      - Export PyTorch models to ONNX with graph verification"
	@echo "  make quantize    - Quantize ONNX models to INT8 using ONNX Runtime"
	@echo "  make test        - Run Python test suite with pytest (53 tests)"
	@echo "  make test-node   - Run Node.js test suite with node:test"
	@echo "  make benchmark   - Run end-to-end latency and throughput benchmarks"
	@echo "  make serve       - Run production FastAPI service with Uvicorn"

install:
	$(PIP) install -e . --no-deps
	$(PIP) install pytest httpx onnx onnxruntime scikit-learn fastapi uvicorn

datasets:
	$(PYTHON) data/generate_datasets.py

train:
	$(PYTHON) training/train_classifier.py
	$(PYTHON) training/train_token_classifier.py
	$(PYTHON) training/train_normalizer.py

export:
	$(PYTHON) training/export_onnx.py

quantize:
	$(PYTHON) training/quantize_onnx.py
	$(PYTHON) training/validate_export.py

test:
	$(PYTHON) -m pytest -v tests/

test-node:
	cd node && npm test

benchmark:
	$(PYTHON) benchmarks/report.py

serve:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
