import os

import psycopg
import pytest

from ingest import db
from recommender import candidates

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running")


@pytest.fixture()
def seeded():
    c = db.connect()
    c.execute("DELETE FROM interactions WHERE user_id='_cand'")
    c.execute("DELETE FROM problems WHERE id IN ('C1','C2','C3')")
    db.upsert_problems(
        c,
        [
            dict(
                id="C1",
                contest_id=1,
                idx="A",
                name="seen",
                rating=1200,
                tags=["dp"],
                solved_count=9,
            ),
            dict(
                id="C2",
                contest_id=2,
                idx="B",
                name="unseen-dp",
                rating=1600,
                tags=["dp"],
                solved_count=8,
            ),
            dict(
                id="C3",
                contest_id=3,
                idx="C",
                name="unseen-math",
                rating=2200,
                tags=["math"],
                solved_count=7,
            ),
        ],
    )
    # user has interacted with C1 only -> it must be excluded
    db.upsert_interaction(
        c,
        dict(
            user_id="_cand",
            problem_id="C1",
            solved=True,
            n_attempts=1,
            first_verdict="OK",
            solved_in_contest=False,
            first_seen_at=None,
            solved_at=None,
        ),
    )
    c.commit()
    yield c
    c.close()


def test_excludes_seen_problems(seeded):
    pids = {x.pid for x in candidates.load_unseen(seeded, "_cand")}
    assert "C1" not in pids and {"C2", "C3"} <= pids


def test_tag_filter(seeded):
    pids = {x.pid for x in candidates.load_unseen(seeded, "_cand", tags=["dp"])}
    assert pids & {"C2", "C3"} == {"C2"}  # only the dp one


def test_b_range_filter(seeded):
    pids = {x.pid for x in candidates.load_unseen(seeded, "_cand", b_range=(1400, 1800))}
    assert pids & {"C2", "C3"} == {"C2"}  # only rating 1600 falls in range
