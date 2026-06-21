import os
import random

import psycopg
import pytest

from ingest import db
from model.irt import SkillModel, TagSkill
from recommender import compose

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


# --- pure interleave tests ---


def test_interleave_avoids_consecutive_when_possible():
    items = [{"topic": t, "pid": f"{t}{i}"} for i, t in enumerate(["a", "a", "b", "c"])]
    out = compose._interleave(items)
    assert compose.max_consecutive_same_tag(out) == 1


def test_interleave_forced_repeat_when_one_topic_dominates():
    items = [{"topic": "a", "pid": str(i)} for i in range(3)] + [{"topic": "b", "pid": "x"}]
    out = compose._interleave(items)
    assert compose.max_consecutive_same_tag(out) == 2  # 3 a's, 1 b -> a,b,a,a unavoidable


# --- DB integration: full daily set ---


@pytest.mark.skipif(not _db_up(), reason="Postgres not running")
def test_daily_set_mixes_modes_interleaves_and_logs():
    c = db.connect()
    c.execute("DELETE FROM recommendations WHERE user_id='_ds'")
    pids = [f"X{t}{b}" for t in ("dp", "gr", "ma") for b in (1300, 1400, 1500, 1570, 1800, 1900)]
    c.execute("DELETE FROM problems WHERE id = ANY(%s)", (pids,))
    rows = []
    for tag, short in (("dp", "dp"), ("greedy", "gr"), ("math", "ma")):
        for b in (1300, 1400, 1500, 1570, 1800, 1900):
            rows.append(
                dict(
                    id=f"X{short}{b}",
                    contest_id=1,
                    idx=f"{short}{b}",
                    name="x",
                    rating=b,
                    tags=[tag],
                    solved_count=100,
                )
            )
    db.upsert_problems(c, rows)
    c.commit()

    m = SkillModel()  # s≈173.7, all gaps 400 vs r_band=1900
    # dp's prereqs satisfied so it's on the frontier (a real user has these populated)
    m.tags["implementation"] = TagSkill(mu=1900, sigma=80)
    m.tags["brute force"] = TagSkill(mu=1900, sigma=80)
    m.tags["dp"] = TagSkill(mu=1500, sigma=300)  # high σ -> Assess
    m.tags["greedy"] = TagSkill(mu=1500, sigma=80)  # low σ (root) -> Train
    m.tags["math"] = TagSkill(mu=1500, sigma=80)  # low σ (root) -> Train

    items = compose.daily_set(c, m, "_ds", r_band=1900, h_hours=21, rng=random.Random(0))

    assert items, "expected a non-empty daily set"
    assert compose.max_consecutive_same_tag(items) <= 1  # 3 topics -> interleavable
    assert any(it["mode"] == "assess" for it in items)  # dp is high-σ
    assert all(0 < it["propensity"] <= 1 for it in items)
    assert all(it["why"] for it in items)

    logged = c.execute(
        "SELECT count(*) AS n, count(propensity) AS p FROM recommendations WHERE user_id='_ds'"
    ).fetchone()
    assert logged["n"] == len(items) and logged["p"] == len(items)  # propensity always set
    c.close()
