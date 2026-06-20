"""Postgres access for ingest (psycopg3, raw SQL — no ORM, ponytail)."""

from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from config import settings


def connect() -> psycopg.Connection:
    return psycopg.connect(settings.database_url, row_factory=dict_row)


def upsert_problems(conn: psycopg.Connection, rows: list[dict]) -> int:
    sql = """
        INSERT INTO problems (id, contest_id, idx, name, rating, tags, solved_count, updated_at)
        VALUES (%(id)s, %(contest_id)s, %(idx)s, %(name)s, %(rating)s, %(tags)s,
                %(solved_count)s, now())
        ON CONFLICT (id) DO UPDATE SET
            contest_id=EXCLUDED.contest_id, idx=EXCLUDED.idx, name=EXCLUDED.name,
            rating=EXCLUDED.rating, tags=EXCLUDED.tags,
            solved_count=EXCLUDED.solved_count, updated_at=now()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def insert_problems_if_absent(conn: psycopg.Connection, rows: list[dict]) -> int:
    """FK-safety insert for problems referenced by submissions but maybe not in the
    catalog. ON CONFLICT DO NOTHING so it never clobbers catalog data (e.g. solved_count)."""
    sql = """
        INSERT INTO problems (id, contest_id, idx, name, rating, tags, solved_count, updated_at)
        VALUES (%(id)s, %(contest_id)s, %(idx)s, %(name)s, %(rating)s, %(tags)s,
                %(solved_count)s, now())
        ON CONFLICT (id) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def upsert_interaction(conn: psycopg.Connection, ep: dict) -> None:
    sql = """
        INSERT INTO interactions
            (user_id, problem_id, solved, n_attempts, first_verdict,
             solved_in_contest, first_seen_at, solved_at)
        VALUES (%(user_id)s, %(problem_id)s, %(solved)s, %(n_attempts)s, %(first_verdict)s,
                %(solved_in_contest)s, %(first_seen_at)s, %(solved_at)s)
        ON CONFLICT (user_id, problem_id) DO UPDATE SET
            solved=EXCLUDED.solved, n_attempts=EXCLUDED.n_attempts,
            first_verdict=EXCLUDED.first_verdict,
            solved_in_contest=EXCLUDED.solved_in_contest,
            first_seen_at=EXCLUDED.first_seen_at, solved_at=EXCLUDED.solved_at
    """
    conn.execute(sql, ep)


def get_interaction(conn: psycopg.Connection, user_id: str, problem_id: str) -> dict | None:
    return conn.execute(
        "SELECT * FROM interactions WHERE user_id=%s AND problem_id=%s",
        (user_id, problem_id),
    ).fetchone()


def get_cursor(conn: psycopg.Connection, user_id: str) -> int:
    row = conn.execute(
        "SELECT last_creation_time FROM ingest_state WHERE user_id=%s", (user_id,)
    ).fetchone()
    return int(row["last_creation_time"]) if row else 0


def set_cursor(conn: psycopg.Connection, user_id: str, last_creation_time: int) -> None:
    conn.execute(
        """
        INSERT INTO ingest_state (user_id, last_creation_time) VALUES (%s, %s)
        ON CONFLICT (user_id) DO UPDATE SET last_creation_time=EXCLUDED.last_creation_time
        """,
        (user_id, last_creation_time),
    )


def catalog_age_seconds(conn: psycopg.Connection) -> float | None:
    row = conn.execute("SELECT max(updated_at) AS m FROM problems").fetchone()
    if not row or row["m"] is None:
        return None
    return (datetime.now(timezone.utc) - row["m"]).total_seconds()
