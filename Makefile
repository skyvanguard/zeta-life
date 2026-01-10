# Zeta-Life Makefile
# ===================
# Common tasks for development and deployment

.PHONY: install test lint docs clean docker reproduce help

# Default target
help:
	@echo "Zeta-Life Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install package in development mode"
	@echo "  test         Run all tests"
	@echo "  lint         Run linters (ruff, mypy)"
	@echo "  docs         Build Sphinx documentation"
	@echo "  reproduce    Reproduce all paper results"
	@echo "  validate     Validate reproduction against expected outputs"
	@echo "  docker       Build Docker image"
	@echo "  clean        Remove build artifacts"
	@echo ""

# Installation
install:
	pip install -e ".[dev,docs,full]"

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src/zeta_life --cov-report=html

# Linting
lint:
	ruff check src/ tests/
	mypy src/zeta_life --ignore-missing-imports

format:
	black src/ tests/ experiments/
	ruff check --fix src/ tests/

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Reproduction
reproduce:
	@echo "Generating paper figures..."
	python scripts/generate_paper_figures.py
	@echo ""
	@echo "Running SYNTH-v2 consolidation..."
	python experiments/consciousness/exp_ipuesa_synth_v2_consolidation.py
	@echo ""
	@echo "Done! Results in results/"

validate:
	python scripts/validate_reproduction.py

# Docker
docker:
	docker build -t zeta-life:latest .

docker-test:
	docker-compose run --rm test

docker-notebook:
	docker-compose up notebook

# Cleaning
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf docs/_build/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Quick demo
quickstart:
	python demos/quickstart.py

# Run interactive chat
chat:
	python demos/chat_psyche.py --reflection
