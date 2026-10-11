# Caïssa-1 — build-phase Makefile
# Targets required by GOAL.md §4.6: data / smoke / eval / lint / test
# Plus: install, format, train (the Definition of Awesome entry point), clean.

SHELL := /bin/bash
PY ?= python
CONFIG ?= configs/tiny.yaml

.DEFAULT_GOAL := help
.PHONY: help install lint format test smoke data data-sample eval train clean ci

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

lint: ## Static checks: ruff + black --check + mypy
	$(PY) -m ruff check src tests tools
	$(PY) -m black --check src tests tools
	$(PY) -m mypy src/caissa

format: ## Auto-format (ruff --fix + black)
	$(PY) -m ruff check --fix src tests tools
	$(PY) -m black src tests tools

test: ## Unit + golden + property tests (fast, no network)
	$(PY) -m pytest -q

smoke: ## CPU micro-benchmark: tiny model, 100 steps, 1000 sampled moves must be 100% legal
	$(PY) -m caissa.training.smoke --config $(CONFIG)

data: ## Show the data manifest status (no download)
	$(PY) -m caissa.acquisition.manifest --status

data-sample: ## Phase 1 DoD: pull a month of Lichess + 1000 puzzles + a TWIC month (resumable)
	$(PY) -m caissa.acquisition.sample --config configs/data-sample.yaml

eval: ## Arena stub + blunder-rate report (mock engine unless STOCKFISH set)
	$(PY) -m caissa.eval.arena --config configs/eval.yaml

train: ## Definition of Awesome: launch Act I on a real cluster
	$(PY) -m caissa.training.act1 --config $(CONFIG)

ci: lint test ## What CI runs on every push

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.part' -delete
