.PHONY: help install install-docs serve site site-strict pdf test test-all lint fmt clean assets smoke scanners precommit

help:
	@echo "Targets:"
	@echo "  install        Install package + core + dev extras"
	@echo "  install-docs   Install everything needed to build the site (+ ml/docs)"
	@echo "  serve          Live-reload MkDocs site at http://127.0.0.1:8000"
	@echo "  site           Build the static HTML site into ./site"
	@echo "  site-strict    Build with --strict (CI uses this)"
	@echo "  pdf            Build the single PDF book at site/pdf/"
	@echo "  test           Run pytest, excluding network tests"
	@echo "  test-all       Run every test, including network-dependent ones"
	@echo "  lint           Ruff + Black --check"
	@echo "  fmt            Format with Black + Ruff --fix"
	@echo "  assets         Regenerate chart assets used in the course"
	@echo "  smoke          End-to-end integration smoke test (synthetic data)"
	@echo "  scanners       Run the daily scanner report against a default universe"
	@echo "  precommit      Install pre-commit hooks"
	@echo "  clean          Remove build artifacts"

install:
	pip install -e ".[core,dev]"

install-docs:
	pip install -e ".[core,ml,docs,dev]"

serve:
	mkdocs serve -a 127.0.0.1:8000

site:
	mkdocs build

site-strict:
	mkdocs build --strict

pdf:
	ENABLE_PDF_EXPORT=1 mkdocs build -f mkdocs-pdf.yml
	@echo "PDF is at: site/pdf/python-engine-course.pdf"

assets:
	python scripts/build_assets.py

smoke:
	python scripts/end_to_end_smoke.py

scanners:
	python scripts/run_scanners.py

test:
	pytest -m "not network"

test-all:
	pytest

lint:
	ruff check engine tests scripts
	black --check engine tests scripts

fmt:
	black engine tests scripts
	ruff check --fix engine tests scripts

precommit:
	pip install pre-commit
	pre-commit install

clean:
	rm -rf site build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
