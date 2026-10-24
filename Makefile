PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PORT ?= 8080
ADMIN_API_KEY ?= change-me-admin
DEMO_API_KEY ?= sk-gw-demo-local-key-0001

.PHONY: venv install dev run test lint fmt openapi traffic docker-up docker-down clean

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

dev: install

run:
	ADMIN_API_KEY=$(ADMIN_API_KEY) DEMO_API_KEY=$(DEMO_API_KEY) $(BIN)/llm-gateway serve --port $(PORT)

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

fmt:
	$(BIN)/ruff format src tests scripts
	$(BIN)/ruff check --fix src tests scripts

openapi:
	$(BIN)/python scripts/export_openapi.py

# Populate the dashboard: run against a gateway started with `make run`.
traffic:
	$(BIN)/python scripts/demo_traffic.py --url http://127.0.0.1:$(PORT) --key $(DEMO_API_KEY) --admin-key $(ADMIN_API_KEY)

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info gateway.db*
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
