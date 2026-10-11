# Caïssa-1 — build-phase Makefile
# Targets required by GOAL.md §4.6: data / smoke / eval / lint / test
# Plus: install, format, train (the Definition of Awesome entry point), clean.
#
# ADR-0005 (environment parity): every target delegates to scripts/run.py, so the
# local loop (Windows, no make/git) and the CI runner converge on ONE source of
# commands. `make X` and `python scripts/run.py X` are the same thing.

SHELL := /bin/bash
PY ?= python
CONFIG ?= configs/tiny.yaml
PGN ?= data/raw/lichess/lichess_day_2013-01.pgn.zst
OUT ?= data/shards
QA ?= data/qa/processing.json
LIMIT ?= 10000
ENGINE ?= auto

.DEFAULT_GOAL := help
.PHONY: help install lint format test smoke data data-sample processing label eval train clean ci

help: ## Show this help
	@$(PY) scripts/run.py help

install: ## Editable install with dev extras
	@$(PY) scripts/run.py install

lint: ## Static checks: ruff + black --check + mypy
	@$(PY) scripts/run.py lint

format: ## Auto-format (ruff --fix + black)
	@$(PY) scripts/run.py format

test: ## Unit + golden + property tests (fast, no network)
	@$(PY) scripts/run.py test

smoke: ## CPU micro-benchmark: tiny model, 100 steps, 1000 sampled moves must be 100% legal
	@$(PY) scripts/run.py smoke --config $(CONFIG)

data: ## Show the data manifest status (no download)
	@$(PY) scripts/run.py data

data-sample: ## Phase 1 DoD: pull a month of Lichess + 1000 puzzles + a TWIC month (resumable)
	@$(PY) scripts/run.py data-sample --config configs/data-sample.yaml

processing: ## Phase 2: stream a PGN -> filter -> dedup -> shard -> QA report
	@$(PY) scripts/run.py processing --pgn $(PGN) --out $(OUT) --qa $(QA)

label: ## Phase 2: blunder labeling (Stockfish when present, else MOCK)
	@$(PY) scripts/run.py label --pgn $(PGN) --limit $(LIMIT) --engine $(ENGINE)

eval: ## Arena stub + blunder-rate report (mock engine unless STOCKFISH set)
	@$(PY) scripts/run.py eval --config configs/eval.yaml

train: ## Definition of Awesome: launch Act I on a real cluster
	@$(PY) scripts/run.py train --config $(CONFIG)

ci: lint test ## What CI runs on every push
	@$(PY) scripts/run.py ci

clean: ## Remove caches and build artifacts
	@$(PY) scripts/run.py clean
