import os
import psycopg
import pytest

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")

EXPECTED = {
    "problems", "interactions", "topic_skill", "reviews",
    "recommendations", "learned_params", "ingest_state",
}


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_up(), reason="Postgres not running (docker compose up -d db)")
def test_all_tables_exist():
    with psycopg.connect(DB) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert EXPECTED <= names


@pytest.mark.skipif(not _db_up(), reason="Postgres not running")
def test_propensity_and_unique_constraint():
    with psycopg.connect(DB) as conn:
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='recommendations'"
        ).fetchall()
        assert "propensity" in {c[0] for c in cols}
        uniq = conn.execute(
            "SELECT conname FROM pg_constraint WHERE conname='interactions_user_problem_uniq'"
        ).fetchall()
        assert uniq, "missing UNIQUE(user_id, problem_id) on interactions"
