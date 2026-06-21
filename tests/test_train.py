import os
import random

import psycopg
import pytest

from ingest import db
from model.irt import SkillModel, TagSkill
from recommender import train
from recommender.candidates import Candidate

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


def _c(pid, b, tags, sc=100):
    return Candidate(pid=pid, b=b, tags=tags, solved_count=sc)


# --- pure tests (no DB) ---


def test_topic_scores_gap_times_weight():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1500, sigma=100)
    sc = train.topic_scores(m, ["dp"], r_band=1900, weights={"dp": 0.5})
    assert abs(sc["dp"] - (1900 - 1500) * 0.5) < 1e-9


def test_topic_scores_zero_gap_when_strong():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=2000, sigma=100)  # already above target
    sc = train.topic_scores(m, ["dp"], r_band=1900, weights={"dp": 1.0})
    assert sc["dp"] == 0.0


def test_sample_topics_no_dups_and_k_and_positive_only():
    rng = random.Random(0)
    scores = {"a": 5.0, "b": 3.0, "c": 0.0}  # c has zero score
    out = train.sample_topics(scores, k=3, rng=rng)
    tags = [t for t, _ in out]
    assert len(tags) == 2 and len(set(tags)) == 2  # only a,b (c excluded), no dups
    assert all(0 < pr <= 1 for _, pr in out)


def test_sample_topics_favours_high_score():
    counts = {"hi": 0, "lo": 0}
    for seed in range(200):
        out = train.sample_topics({"hi": 99.0, "lo": 1.0}, k=1, rng=random.Random(seed))
        counts[out[0][0]] += 1
    assert counts["hi"] > counts["lo"] * 5  # strongly prefers the high-score topic


def test_pick_in_band_only_in_band():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1500, sigma=100)
    # P(b=1500)=0.5 (out of 0.55-0.80), P(b=1300)~0.76 (in band), P(b=900)~0.96 (out)
    cands = [_c("easy", 900, ["dp"]), _c("inband", 1300, ["dp"]), _c("hard", 1500, ["dp"])]
    pick, prop = train.pick_in_band(m, cands, random.Random(0), band=(0.55, 0.80))
    assert pick.pid == "inband" and 0 < prop <= 1


def test_pick_in_band_none_when_empty():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1500, sigma=100)
    pick, prop = train.pick_in_band(
        m, [_c("hard", 2500, ["dp"])], random.Random(0), band=(0.55, 0.80)
    )
    assert pick is None and prop == 0.0


# --- DB test ---


@pytest.mark.skipif(not _db_up(), reason="Postgres not running")
def test_goal_weights_normalized_frequency():
    c = db.connect()
    c.execute("DELETE FROM problems WHERE id IN ('G1','G2','G3','G4')")
    # Use tags unique to this test so the catalog-frequency count is not polluted by
    # other tests' seeded problems sharing real tags (e.g. dp/math) in the same band.
    db.upsert_problems(
        c,
        [
            dict(
                id="G1",
                contest_id=1,
                idx="A",
                name="x",
                rating=1900,
                tags=["gw_alpha"],
                solved_count=1,
            ),
            dict(
                id="G2",
                contest_id=1,
                idx="B",
                name="x",
                rating=1850,
                tags=["gw_alpha"],
                solved_count=1,
            ),
            dict(
                id="G3",
                contest_id=1,
                idx="C",
                name="x",
                rating=1950,
                tags=["gw_beta"],
                solved_count=1,
            ),
            dict(
                id="G4",
                contest_id=1,
                idx="D",
                name="x",
                rating=600,
                tags=["gw_alpha"],
                solved_count=1,
            ),  # out of band
        ],
    )
    c.commit()
    w = train.goal_weights(c, ["gw_alpha", "gw_beta"], r_band=1900, band=200)
    # in-band: gw_alpha has 2 (G1,G2), gw_beta has 1 (G3); G4 excluded (rating 600)
    assert abs(w["gw_alpha"] - 2 / 3) < 1e-9 and abs(w["gw_beta"] - 1 / 3) < 1e-9
    c.close()
