# Zeta-Life Makefile
# ===================
# Common tasks for development and deployment

.PHONY: install test test-cov lint format clean docker docker-test reproduce quickstart help

# Default target
help:
	@echo "Zeta-Life Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install package in development mode"
	@echo "  quickstart   Run the 60-line Conscious Kernel demo"
	@echo "  test         Run all tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run linters (ruff, mypy)"
	@echo "  format       Format code (black, ruff --fix)"
	@echo "  reproduce    Re-run the headline kernel experiments"
	@echo "  docker       Build Docker image"
	@echo "  clean        Remove build artifacts"
	@echo ""

# Installation
install:
	pip install -e ".[dev,full]"

# Quick demo
quickstart:
	PYTHONPATH=src python demos/quickstart.py

# Testing
test:
	PYTHONPATH=src pytest tests/ -v --tb=short

test-cov:
	PYTHONPATH=src pytest tests/ -v --cov=src/zeta_life --cov-report=html

# Linting
lint:
	ruff check src/ tests/
	mypy src/zeta_life --ignore-missing-imports

format:
	black src/ tests/ experiments/
	ruff check --fix src/ tests/

# Reproduction — the live kernel experiments behind the paper's headline results
reproduce:
	@echo "Kernel validation (full active-inference cycle)..."
	PYTHONPATH=src python experiments/kernel/exp_conscious_kernel_validation.py
	@echo ""
	@echo "Multi-kernel organism emergence..."
	PYTHONPATH=src python experiments/kernel/exp_organism_emergence.py
	@echo ""
	@echo "Psi on real datasets..."
	PYTHONPATH=src python experiments/datasets/exp_real_data_psi.py
	@echo ""
	@echo "Done! Results in results/"

# Docker
docker:
	docker build -t zeta-life:latest .

docker-test:
	docker-compose run --rm test

# Cleaning
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
