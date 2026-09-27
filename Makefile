.PHONY: setup check

setup:
	uv sync --locked
	uv run pre-commit install

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked mypy src