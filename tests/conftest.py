"""Test isolation: integration tests run against a dedicated `whetstone_test`
database so they never touch dev/ingest data (which would pollute the catalog
freshness check and collide with real CF problem ids).

Pointing DATABASE_URL at the test DB *before* config/ingest import means
config.settings and every `db.connect()` pick it up (env var overrides .env).
"""
import os
import subprocess

import psycopg

ADMIN_URL = "postgresql://whetstone:whetstone@localhost:5432/whetstone"
TEST_URL = "postgresql://whetstone:whetstone@localhost:5432/whetstone_test"

# Must happen at import time, before any test module imports config/ingest.
os.environ["DATABASE_URL"] = TEST_URL


def _pg_up(url: str) -> bool:
    try:
        psycopg.connect(url, connect_timeout=2).close()
        return True
    except Exception:
        return False


def pytest_configure(config):
    """Create + migrate the test database once per session (if Postgres is up)."""
    if not _pg_up(ADMIN_URL):
        return  # DB down → integration tests skip via their own guards
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        exists = admin.execute(
            "SELECT 1 FROM pg_database WHERE datname='whetstone_test'"
        ).fetchone()
        if not exists:
            admin.execute("CREATE DATABASE whetstone_test")
    subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        env={**os.environ, "DATABASE_URL": TEST_URL},
        check=True,
    )
