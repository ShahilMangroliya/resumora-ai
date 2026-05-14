.PHONY: install dev test lint

install:
	uv sync --all-packages
	npm --prefix apps/web install

dev:
	uv run honcho start

test:
	uv run pytest

lint:
	uv run ruff check .
