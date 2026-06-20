import os
from datetime import datetime, timezone
import psycopg
import pytest
from ingest import db

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running")


@pytest.fixture()
def conn():
    c = db.connect()
    c.execute("DELETE FROM interactions WHERE user_id='_test'")
    c.execute("DELETE FROM problems WHERE id='9999Z'")
    c.execute("DELETE FROM ingest_state WHERE user_id='_test'")
    c.commit()
    yield c
    c.close()


def test_upsert_problem_is_idempotent(conn):
    row = dict(id="9999Z", contest_id=9999, idx="Z", name="t", rating=800,
               tags=["math"], solved_count=10)
    assert db.upsert_problems(conn, [row]) == 1
    row["solved_count"] = 20
    db.upsert_problems(conn, [row])
    conn.commit()
    got = conn.execute("SELECT solved_count FROM problems WHERE id='9999Z'").fetchone()
    assert got["solved_count"] == 20


def test_upsert_interaction_and_get(conn):
    db.upsert_problems(conn, [dict(id="9999Z", contest_id=9999, idx="Z", name="t",
                                   rating=800, tags=["math"], solved_count=10)])
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ep = dict(user_id="_test", problem_id="9999Z", solved=True, n_attempts=2,
              first_verdict="WRONG_ANSWER", solved_in_contest=False,
              first_seen_at=ts, solved_at=ts)
    db.upsert_interaction(conn, ep)
    conn.commit()
    got = db.get_interaction(conn, "_test", "9999Z")
    assert got["solved"] is True and got["n_attempts"] == 2


def test_cursor_roundtrip(conn):
    assert db.get_cursor(conn, "_test") == 0
    db.set_cursor(conn, "_test", 1700000000)
    conn.commit()
    assert db.get_cursor(conn, "_test") == 1700000000
