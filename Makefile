# ── World Cup 2026 Sentiment Analysis — Makefile ──────────────────────────
# Cross-platform targets (Windows / Unix) using python -m where possible.
# Uses PowerShell as the recipe shell for reliable Windows execution.

SHELL        = powershell.exe
.SHELLFLAGS  = -NoProfile -Command

.PHONY: help setup collect refresh dashboard test lint

# ── OS-agnostic venv paths ─────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    VENV_PYTHON     = venv\Scripts\python
    VENV_PIP        = venv\Scripts\python -m pip
    VENV_PRE_COMMIT = venv\Scripts\pre-commit
    VENV_SPACY      = venv\Scripts\python -m spacy
else
    VENV_PYTHON     = venv/bin/python
    VENV_PIP        = venv/bin/python -m pip
    VENV_PRE_COMMIT = venv/bin/pre-commit
    VENV_SPACY      = venv/bin/python -m spacy
endif

help:
	@echo "World Cup 2026 Sentiment Analysis — Makefile"
	@echo ""
	@echo "  setup       Create venv, install requirements, download spaCy, install pre-commit"
	@echo "  collect     Run manual collection: python -m src.pipeline --step collect"
	@echo "  refresh     Pull data + execute notebooks 02-03-04 + commit results"
	@echo "  dashboard   Launch Streamlit dashboard"
	@echo "  test        Run pytest tests/ -v"
	@echo "  lint        Run pre-commit run --all-files"

setup:
	python -m venv venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	$(VENV_SPACY) download es_core_news_sm
	$(VENV_SPACY) download en_core_web_sm
	$(VENV_PRE_COMMIT) install
	@echo "Setup complete."

collect:
	python -m src.pipeline --step collect

refresh:
	python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "notebooks/02_limpieza_preprocesamiento.ipynb"
	python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "notebooks/03_analisis_sentimiento.ipynb"
	python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "notebooks/04_topic_modeling_ner.ipynb"
	git add notebooks/
	git commit -m "data: refresh pipeline $$(Get-Date -Format 'yyyy-MM-dd')"
	git push origin main

dashboard:
	streamlit run dashboard/app.py

test:
	python -m pytest tests/ -v

lint:
	pre-commit run --all-files
