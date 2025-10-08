.PHONY: help test test-verbose test-coverage clean lint format install dev-install

PYTHON = .venv/bin/python
PIP = .venv/bin/pip
PYTEST = .venv/bin/pytest

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(PIP) install -r requirements.txt

dev-install:  ## Install development dependencies
	$(PIP) install -e ".[dev]"

test:  ## Run tests
	$(PYTHON) -m unittest discover tests/ -v

test-pytest:  ## Run tests with pytest
	$(PYTEST) tests/ -v

test-coverage:  ## Run tests with coverage
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

test-verbose:  ## Run tests with verbose output
	$(PYTHON) -m unittest discover tests/ -v

lint:  ## Run linting
	$(PYTHON) -m flake8 src/ tests/

format:  ## Format code
	$(PYTHON) -m black src/ tests/

type-check:  ## Run type checking
	$(PYTHON) -m mypy src/

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

run-notebooks:  ## Run Jupyter notebooks
	$(PYTHON) -m jupyter lab notebooks/

build:  ## Build the package
	$(PYTHON) -m build

all-checks: lint type-check test  ## Run all quality checks