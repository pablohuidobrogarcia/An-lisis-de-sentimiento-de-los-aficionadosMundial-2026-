# ── World Cup 2026 Sentiment Analysis — Makefile ──────────────────────────
# Common tasks for setup, pipeline execution, testing, and formatting.

.PHONY: help setup pipeline dashboard test lint format clean

help:
	@echo "World Cup 2026 Sentiment Analysis — Makefile"
	@echo ""
	@echo "  setup            Create venv, install dependencies, download spacy models"
	@echo "  pipeline         Run the full end-to-end pipeline"
	@echo "  pipeline-quick   Run pipeline using cached collection (skip collect)"
	@echo "  dashboard        Launch Streamlit dashboard"
	@echo "  test             Run pytest with coverage"
	@echo "  lint             Run ruff linter"
	@echo "  format           Format code with black + isort"
	@echo "  clean            Remove cache files and build artifacts"
	@echo "  pre-commit       Install pre-commit hooks"

setup:
	python -m venv venv
	. venv/bin/activate || venv\Scripts\activate
	pip install --upgrade pip
	pip install -r requirements.txt
	python -m spacy download es_core_news_sm
	python -m spacy download en_core_web_sm
	cp -n .env.example .env || echo ".env already exists — edit it with your API keys"

pipeline:
	python -m src.pipeline

pipeline-quick:
	python -m src.pipeline --skip-collect

dashboard:
	streamlit run dashboard/app.py

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ dashboard/ evaluation/

format:
	black src/ tests/ dashboard/ evaluation/
	isort src/ tests/ dashboard/ evaluation/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info dist build

pre-commit:
	pre-commit install
