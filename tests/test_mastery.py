import os

import psycopg
import pytest

from config import MASTERY_MIN_STABILITY
from ingest import db
from model import mastery
from model.irt import SkillModel, TagSkill
from recommender import prereq_dag

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


GOOD_STAB = MASTERY_MIN_STABILITY + 10


def _model():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=2000, sigma=100)  # strong + confident
    return m


def test_mastery_needs_all_three_conditions():
    m = _model()
    rv = {"dp": {"stability": GOOD_STAB}}
    assert mastery.is_mastered(m, rv, "dp", r_band=1900) is True
    # fail μ
    assert mastery.is_mastered(_model_with(1800, 100), rv, "dp", 1900) is False
    # fail σ
    assert mastery.is_mastered(_model_with(2000, 200), rv, "dp", 1900) is False
    # fail stability
    assert mastery.is_mastered(m, {"dp": {"stability": 1.0}}, "dp", 1900) is False


def _model_with(mu, sigma):
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=mu, sigma=sigma)
    return m


def test_frontier_uses_mastery_when_provided():
    m = SkillModel()
    m.tags["math"] = TagSkill(mu=1000, sigma=100)  # weak, but...
    # number theory needs math; with math in `mastered`, it opens despite low μ
    fr = prereq_dag.frontier(m, ["math", "number theory"], 1900, 200, mastered={"math"})
    assert "number theory" in fr
    fr2 = prereq_dag.frontier(m, ["math", "number theory"], 1900, 200, mastered=set())
    assert "number theory" not in fr2  # math not mastered -> gated


@pytest.mark.skipif(not _db_up(), reason="Postgres not running")
def test_mark_mastery_updates_topic_skill():
    c = db.connect()
    c.execute("DELETE FROM topic_skill WHERE user_id='_mast'")
    c.execute("""INSERT INTO topic_skill (user_id, tag, mu, sigma, mastered, snapshot_at)
           VALUES ('_mast','dp',2000,100,false, now()), ('_mast','math',1500,100,false, now())""")
    c.commit()
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=2000, sigma=100)  # mastered (with stability)
    m.tags["math"] = TagSkill(mu=1500, sigma=100)  # not (μ below band)
    rv = {"dp": {"stability": GOOD_STAB}, "math": {"stability": GOOD_STAB}}
    n = mastery.mark_mastery(c, "_mast", m, rv, r_band=1900)
    c.commit()
    assert n == 1
    rows = c.execute("SELECT tag, mastered FROM topic_skill WHERE user_id='_mast'").fetchall()
    got = {r["tag"]: r["mastered"] for r in rows}
    assert got["dp"] is True and got["math"] is False
    c.close()
