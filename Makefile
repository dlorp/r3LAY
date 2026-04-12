# r3LAY development workflow
# Usage: make check (before pushing), make test, make lint, make fmt

VENV := .venv/bin
PYTHON := $(VENV)/python
RUFF := $(VENV)/ruff
PYTEST := $(PYTHON) -m pytest

.PHONY: check test lint fmt serve watch help

## Full pre-push check: lint + format + tests
check: lint fmt test
	@echo "All checks passed."

## Run the test suite
test:
	$(PYTEST) tests/ -v

## Quick test (no verbose, fail-fast)
test-quick:
	$(PYTEST) tests/ -x -q

## Lint check (no auto-fix)
lint:
	$(RUFF) check .

## Format check (no auto-fix)
fmt:
	$(RUFF) format --check .

## Auto-fix lint + format issues
fix:
	$(RUFF) check --fix .
	$(RUFF) format .

## Start the bridge (port 8765)
serve:
	$(PYTHON) -m r3lay.bridge

## Start the file watcher
watch:
	$(PYTHON) -m r3lay.sync

## Help
help:
	@echo "r3LAY development commands:"
	@echo "  make check      Full pre-push check (lint + format + tests)"
	@echo "  make test       Run test suite (verbose)"
	@echo "  make test-quick Quick test (fail-fast, quiet)"
	@echo "  make lint       Lint check"
	@echo "  make fmt        Format check"
	@echo "  make fix        Auto-fix lint + format"
	@echo "  make serve      Start bridge on :8765"
	@echo "  make watch      Start file watcher"
