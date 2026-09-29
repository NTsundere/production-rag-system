.PHONY: help install dev test lint format run docker-build docker-run ingest eval clean

help:
	@echo "install      Install production deps"
	@echo "dev          Install dev deps"
	@echo "test         Run pytest with coverage"
	@echo "lint         Run flake8 + black check"
	@echo "format       Format with black"
	@echo "run          Run FastAPI locally"
	@echo "docker-run   Run via docker-compose"
	@echo "ingest       Ingest a document (FILE=path)"
	@echo "eval         Run RAGAS evaluation"
	@echo "clean        Clean caches"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ --cov=app --cov-report=term-missing

lint:
	flake8 app/ tests/ scripts/
	black --check app/ tests/ scripts/

format:
	black app/ tests/ scripts/

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-run:
	docker-compose up --build

ingest:
	python scripts/ingest.py --file $(FILE)

eval:
	python scripts/run_eval.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete