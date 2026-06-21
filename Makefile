.PHONY: dbup dbdown migrate ingest test fmt lint

dbup:
	docker compose -f infra/docker-compose.yml up -d db

dbdown:
	docker compose -f infra/docker-compose.yml down

migrate:
	uv run alembic upgrade head

ingest:
	uv run python -m ingest.poller

eval:
	uv run python -m eval.run_m1

train:
	uv run python -m model.snapshot

test:
	uv run pytest -q

fmt:
	uv run black . && uv run ruff check --fix .

lint:
	uv run ruff check . && uv run black --check .
