import json
import os

import psycopg
import pytest

from ingest import db
from model import snapshot
from model.irt import SkillModel, TagSkill

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running")


def test_save_snapshot_writes_topic_skill():
    c = db.connect()
    c.execute("DELETE FROM topic_skill WHERE user_id='_snap'")
    c.commit()
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1900, sigma=120)
    m.tags["math"] = TagSkill(mu=1700, sigma=90)
    n = snapshot.save_snapshot(c, "_snap", m)
    c.commit()
    assert n == 2
    rows = c.execute("SELECT tag, mu FROM topic_skill WHERE user_id='_snap'").fetchall()
    assert {r["tag"] for r in rows} == {"dp", "math"}
    c.close()


def test_dump_radar_writes_json(tmp_path):
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1900, sigma=120)
    path = tmp_path / "skills.json"
    snapshot.dump_radar(m, str(path))
    data = json.loads(path.read_text())
    assert data[0]["tag"] == "dp" and data[0]["mu"] == 1900
