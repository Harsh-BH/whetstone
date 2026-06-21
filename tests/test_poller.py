import os
import psycopg
import pytest
from ingest import poller, db
from ingest.cf_client import CFSubmission, CFProblem

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running")


class FakeClient:
    """Stand-in for CFClient with a scripted submission stream."""

    def __init__(self, subs, solved_counts=None):
        self._subs = subs
        self._solved = solved_counts or {}

    def problemset_problems(self):
        return [
            CFProblem.model_validate(
                {"contestId": 1, "index": "A", "name": "p", "rating": 800, "tags": ["math"]}
            )
        ], {"1A": 5000}

    def user_status(self, handle):
        return self._subs


def _sub(id, t, verdict, ptype, cid=1, idx="A"):
    return CFSubmission.model_validate(
        {
            "id": id,
            "creationTimeSeconds": t,
            "verdict": verdict,
            "author": {"participantType": ptype},
            "problem": {
                "contestId": cid,
                "index": idx,
                "name": "p",
                "rating": 800,
                "tags": ["math"],
            },
        }
    )


@pytest.fixture(autouse=True)
def clean():
    c = db.connect()
    c.execute("DELETE FROM interactions WHERE user_id='_pt'")
    c.execute("DELETE FROM ingest_state WHERE user_id='_pt'")
    c.commit()
    c.close()


def test_first_run_lands_episodes():
    client = FakeClient([_sub(1, 10, "WRONG_ANSWER", "PRACTICE"), _sub(2, 20, "OK", "PRACTICE")])
    report = poller.run(handle="_pt", client=client)
    assert report["new_episodes"] == 1
    assert report["new_solves"] == 1
    assert report["cursor"] == 20
    c = db.connect()
    ep = db.get_interaction(c, "_pt", "1A")
    assert ep["solved"] is True and ep["n_attempts"] == 2
    c.close()


def test_second_run_is_incremental_and_merges():
    poller.run(handle="_pt", client=FakeClient([_sub(1, 10, "WRONG_ANSWER", "PRACTICE")]))
    report = poller.run(
        handle="_pt",
        client=FakeClient(
            [
                _sub(1, 10, "WRONG_ANSWER", "PRACTICE"),  # <= cursor, skipped
                _sub(2, 30, "OK", "CONTESTANT"),  # > cursor, merged
            ]
        ),
    )
    c = db.connect()
    ep = db.get_interaction(c, "_pt", "1A")
    assert ep["solved"] is True
    assert ep["n_attempts"] == 2
    assert ep["solved_in_contest"] is True
    assert report["cursor"] == 30
    c.close()


def test_empty_handle_raises(monkeypatch):
    # No handle anywhere (arg empty AND config empty) -> must raise.
    monkeypatch.setattr(poller.settings, "cf_handle", "")
    with pytest.raises(ValueError):
        poller.run(handle="", client=FakeClient([]))
