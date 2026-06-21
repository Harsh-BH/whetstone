import os

import psycopg
import pytest

from eval import rating
from ingest import db
from model.irt import SkillModel, TagSkill

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running")


class _FakeRating:
    def __init__(self, contest_id, new_rating, update_time):
        self.contest_id, self.new_rating, self.update_time = contest_id, new_rating, update_time


class _FakeClient:
    def user_rating(self, handle):
        return [_FakeRating(1, 1500, 1000), _FakeRating(2, 1739, 2000)]


def test_ingest_ratings_and_latest():
    c = db.connect()
    c.execute("DELETE FROM ratings WHERE user_id='_rt'")
    c.commit()
    n = rating.ingest_ratings(c, _FakeClient(), "_rt")
    c.commit()
    assert n == 2
    assert rating.latest_actual(c, "_rt") == 1739  # most recent by update_time
    c.close()


def test_predict_rating_between_min_and_max_mu():
    c = db.connect()
    c.execute("DELETE FROM problems WHERE id IN ('R1','R2')")
    db.upsert_problems(
        c,
        [
            dict(
                id="R1",
                contest_id=1,
                idx="A",
                name="x",
                rating=1900,
                tags=["rt_alpha"],
                solved_count=1,
            ),
            dict(
                id="R2",
                contest_id=1,
                idx="B",
                name="x",
                rating=1900,
                tags=["rt_beta"],
                solved_count=1,
            ),
        ],
    )
    c.commit()
    m = SkillModel()
    m.tags["rt_alpha"] = TagSkill(mu=1600, sigma=100)
    m.tags["rt_beta"] = TagSkill(mu=2000, sigma=100)
    pred = rating.predict_rating(c, m, r_band=1900)
    assert 1600 <= pred <= 2000  # a weighted mean of the two μ
    c.close()


def test_predicted_vs_actual():
    c = db.connect()
    c.execute("DELETE FROM ratings WHERE user_id='_rt2'")
    c.execute("INSERT INTO ratings VALUES ('_rt2', 1, 1700, 100)")
    c.commit()
    m = SkillModel()
    m.tags["rt_alpha"] = TagSkill(mu=1750, sigma=100)
    res = rating.predicted_vs_actual(c, m, "_rt2", r_band=1900, tol=300)
    assert res["actual"] == 1700 and res["abs_error"] is not None
    assert res["tracks"] is True  # |1750-1700| = 50 <= 300
    c.close()
