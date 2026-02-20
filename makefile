start:
	@fastapi dev app/main.py

worker:
	@uv run python -m app.workers.pipeline_worker

migrate-create:
ifndef message
	@read -p "Migration message: " msg && alembic revision --autogenerate -m "$$msg"
else
	@alembic revision --autogenerate -m "$(message)"
endif

migrate:
	@alembic upgrade head

typecheck:
	@uv run python scripts/generate_model_dtos.py
	@uv run python scripts/check_typed_writes.py
	@uv run --extra dev ty check app

generate-dtos:
	@uv run python scripts/generate_model_dtos.py

check-typed-writes:
	@uv run python scripts/check_typed_writes.py

refresh-models:
	@uv run python scripts/refresh_model_selection.py --env development --env staging --env production
