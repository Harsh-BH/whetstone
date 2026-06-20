import json
import httpx
import pytest
from ingest import cf_client


def _resp(payload, status=200):
    return httpx.Response(status, content=json.dumps(payload))


def test_problemset_parsing():
    def handler(request):
        return _resp({"status": "OK", "result": {
            "problems": [{"contestId": 1850, "index": "A", "name": "P", "rating": 800,
                          "tags": ["math"]}],
            "problemStatistics": [{"contestId": 1850, "index": "A", "solvedCount": 1234}],
        }})

    c = cf_client.CFClient(transport=httpx.MockTransport(handler), min_interval=0)
    problems, solved = c.problemset_problems()
    assert problems[0].pid == "1850A"
    assert solved["1850A"] == 1234


def test_user_status_paginates_until_short_page():
    pages = [
        [{"id": 3, "creationTimeSeconds": 30, "verdict": "OK",
          "author": {"participantType": "PRACTICE"},
          "problem": {"contestId": 1, "index": "A", "name": "a", "rating": 800, "tags": []}},
         {"id": 2, "creationTimeSeconds": 20, "verdict": "WRONG_ANSWER",
          "author": {"participantType": "CONTESTANT"},
          "problem": {"contestId": 1, "index": "A", "name": "a", "rating": 800, "tags": []}}],
        [{"id": 1, "creationTimeSeconds": 10, "verdict": "OK",
          "author": {"participantType": "CONTESTANT"},
          "problem": {"contestId": 2, "index": "B", "name": "b", "rating": 900, "tags": ["dp"]}}],
    ]

    def handler(request):
        frm = int(request.url.params["from"])
        page = pages[0] if frm == 1 else pages[1]
        return _resp({"status": "OK", "result": page})

    c = cf_client.CFClient(transport=httpx.MockTransport(handler), min_interval=0, page_size=2)
    subs = c.user_status("x")
    assert [s.id for s in subs] == [3, 2, 1]
    assert subs[1].participant_type == "CONTESTANT"
    assert subs[2].problem.pid == "2B"


def test_backoff_then_success(monkeypatch):
    monkeypatch.setattr(cf_client.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp({"status": "FAILED", "comment": "limit exceeded"}, status=503)
        return _resp({"status": "OK", "result": []})

    c = cf_client.CFClient(transport=httpx.MockTransport(handler), min_interval=0)
    assert c.user_status("x") == []
    assert calls["n"] == 2


def test_failed_status_raises(monkeypatch):
    monkeypatch.setattr(cf_client.time, "sleep", lambda *_: None)

    def handler(request):
        return _resp({"status": "FAILED", "comment": "handle not found"}, status=400)

    c = cf_client.CFClient(transport=httpx.MockTransport(handler), min_interval=0, max_retries=2)
    with pytest.raises(cf_client.CFError):
        c.user_info("nope")
