import os
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from ingest import db
from retention import scheduler

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running")

NOW = datetime(2026, 6, 21, tzinfo=timezone.utc)


@pytest.fixture()
def seeded():
    c = db.connect()
    c.execute("DELETE FROM interactions WHERE user_id='_sch'")
    c.execute("DELETE FROM reviews WHERE user_id='_sch'")
    c.execute("DELETE FROM problems WHERE id IN ('OLD1','FRESH1')")
    db.upsert_problems(
        c,
        [
            dict(
                id="OLD1",
                contest_id=1,
                idx="A",
                name="x",
                rating=1500,
                tags=["oldtag"],
                solved_count=1,
            ),
            dict(
                id="FRESH1",
                contest_id=1,
                idx="B",
                name="x",
                rating=1500,
                tags=["freshtag"],
                solved_count=1,
            ),
        ],
    )
    old = NOW - timedelta(days=120)
    db.upsert_interaction(
        c,
        dict(
            user_id="_sch",
            problem_id="OLD1",
            solved=True,
            n_attempts=1,
            first_verdict="OK",
            solved_in_contest=False,
            first_seen_at=old,
            solved_at=old,
        ),
    )
    db.upsert_interaction(
        c,
        dict(
            user_id="_sch",
            problem_id="FRESH1",
            solved=True,
            n_attempts=1,
            first_verdict="OK",
            solved_in_contest=False,
            first_seen_at=NOW,
            solved_at=NOW,
        ),
    )
    c.commit()
    yield c
    c.close()


def test_cold_start_state_per_tag(seeded):
    state = scheduler.cold_start_state(seeded, "_sch")
    assert {"oldtag", "freshtag"} <= set(state)
    assert state["oldtag"]["stability"] >= 1.0
    assert state["oldtag"]["last_review"].date() == (NOW - timedelta(days=120)).date()


def test_due_queue_surfaces_decayed_excludes_fresh(seeded):
    scheduler.save_reviews(seeded, "_sch", scheduler.cold_start_state(seeded, "_sch"))
    seeded.commit()
    q = scheduler.due_queue(seeded, "_sch", now=NOW)
    concepts = [d["concept"] for d in q]
    assert "oldtag" in concepts  # solved 120d ago, S=1 -> decayed
    assert "freshtag" not in concepts  # solved today -> still fresh
    # most-decayed first
    assert q == sorted(q, key=lambda d: d["retrievability"])
