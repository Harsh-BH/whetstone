from ingest.cf_client import CFSubmission
from ingest import normalize


def sub(id, t, verdict, ptype, cid=1, idx="A"):
    return CFSubmission.model_validate({
        "id": id, "creationTimeSeconds": t, "verdict": verdict,
        "author": {"participantType": ptype},
        "problem": {"contestId": cid, "index": idx, "name": "p", "rating": 800, "tags": ["math"]},
    })


def test_solved_after_wa_practice():
    subs = [sub(1, 10, "WRONG_ANSWER", "PRACTICE"), sub(2, 20, "OK", "PRACTICE")]
    eps = normalize.normalize("u", subs)
    assert len(eps) == 1
    e = eps[0]
    assert e.problem_id == "1A"
    assert e.solved is True
    assert e.n_attempts == 2
    assert e.first_verdict == "WRONG_ANSWER"
    assert e.solved_in_contest is False
    assert e.first_seen_at == 10
    assert e.solved_at == 20


def test_solved_first_try_in_contest():
    eps = normalize.normalize("u", [sub(1, 5, "OK", "CONTESTANT")])
    e = eps[0]
    assert e.n_attempts == 1 and e.first_verdict == "OK"
    assert e.solved_in_contest is True and e.solved_at == 5


def test_unsolved_only_failures():
    eps = normalize.normalize("u", [sub(1, 1, "WRONG_ANSWER", "PRACTICE"),
                                    sub(2, 2, "TIME_LIMIT_EXCEEDED", "PRACTICE")])
    e = eps[0]
    assert e.solved is False and e.solved_at is None
    assert e.n_attempts == 2 and e.solved_in_contest is False


def test_compilation_error_is_ignored():
    eps = normalize.normalize("u", [sub(1, 1, "COMPILATION_ERROR", "PRACTICE")])
    assert eps == []


def test_two_problems_two_episodes():
    eps = normalize.normalize("u", [sub(1, 1, "OK", "PRACTICE", cid=1, idx="A"),
                                    sub(2, 2, "OK", "PRACTICE", cid=2, idx="B")])
    assert {e.problem_id for e in eps} == {"1A", "2B"}


def test_problem_without_contest_id_skipped():
    s = CFSubmission.model_validate({
        "id": 1, "creationTimeSeconds": 1, "verdict": "OK",
        "author": {"participantType": "PRACTICE"},
        "problem": {"index": "A", "name": "acmsguru", "tags": []},
    })
    assert normalize.normalize("u", [s]) == []


def test_merge_unsolved_then_solved():
    old = normalize.normalize("u", [sub(1, 10, "WRONG_ANSWER", "PRACTICE")])[0]
    new = normalize.normalize("u", [sub(2, 20, "OK", "CONTESTANT")])[0]
    m = normalize.merge(old, new)
    assert m.solved is True and m.solved_at == 20
    assert m.n_attempts == 2
    assert m.first_verdict == "WRONG_ANSWER"
    assert m.first_seen_at == 10
    assert m.solved_in_contest is True


def test_merge_none_old_returns_new():
    new = normalize.normalize("u", [sub(1, 1, "OK", "PRACTICE")])[0]
    assert normalize.merge(None, new) is new
