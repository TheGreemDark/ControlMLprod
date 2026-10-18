.PHONY: setup check s3-up s3-down ingest data

setup:
	uv sync --locked
	uv run pre-commit install

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked mypy src

s3-up:
	docker compose up -d --wait

s3-down:
	docker compose down

ingest:
	uv run python -m controlmlprod.data.ingest

data:
	uv run python -m controlmlprod.data.pipeline
