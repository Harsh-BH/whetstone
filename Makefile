.PHONY: dbup dbdown migrate ingest eval train serve serve-web test fmt lint

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

serve:
	uv run uvicorn api.app:app --port 8000 --reload

serve-web:
	cd web && npm run dev

test:
	uv run pytest -q

fmt:
	uv run black . && uv run ruff check --fix .

lint:
	uv run ruff check . && uv run black --check .
