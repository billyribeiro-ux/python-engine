.PHONY: help install install-docs serve site pdf test lint fmt clean assets

help:
	@echo "Targets:"
	@echo "  install        Install package + core extras (no ML deps)"
	@echo "  install-docs   Install everything needed to build the site"
	@echo "  serve          Live-reload MkDocs site at http://127.0.0.1:8000"
	@echo "  site           Build the static HTML site into ./site"
	@echo "  pdf            Build the single PDF book into ./site"
	@echo "  test           Run pytest"
	@echo "  lint           Ruff + Black --check"
	@echo "  fmt            Format with Black + Ruff --fix"
	@echo "  assets         Regenerate chart assets used in the course"
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
	ENABLE_PDF_EXPORT=1 mkdocs build
	@echo "PDF is at: site/pdf/python-engine-course.pdf"

assets:
	python scripts/build_assets.py

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

clean:
	rm -rf site build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
