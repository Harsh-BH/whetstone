"""Collapse CF submissions into per-(user,problem) episodes (docs/06).

Incremental-merge correctness: CF submissions are chronological and the
ingest cursor guarantees every NEW submission is later than every stored
one, so merge(old, new) needs no re-fetch of history.
"""
from dataclasses import dataclass

from ingest.cf_client import CFSubmission

SOLVE_VERDICT = "OK"
FAIL_VERDICTS = {"WRONG_ANSWER", "TIME_LIMIT_EXCEEDED", "RUNTIME_ERROR", "MEMORY_LIMIT_EXCEEDED"}
GRADED = {SOLVE_VERDICT} | FAIL_VERDICTS
IN_CONTEST = {"CONTESTANT", "MANAGER"}


@dataclass
class Episode:
    user_id: str
    problem_id: str
    solved: bool
    n_attempts: int
    first_verdict: str
    solved_in_contest: bool
    first_seen_at: int
    solved_at: int | None


def normalize(user_id: str, submissions: list[CFSubmission]) -> list[Episode]:
    by_problem: dict[str, list[CFSubmission]] = {}
    for s in submissions:
        if s.verdict not in GRADED or s.problem.contest_id is None:
            continue
        by_problem.setdefault(s.problem.pid, []).append(s)

    episodes: list[Episode] = []
    for pid, subs in by_problem.items():
        subs.sort(key=lambda s: s.creation_time)
        first = subs[0]
        solve = next((s for s in subs if s.verdict == SOLVE_VERDICT), None)
        episodes.append(Episode(
            user_id=user_id,
            problem_id=pid,
            solved=solve is not None,
            n_attempts=len(subs),
            first_verdict=first.verdict or "",
            solved_in_contest=bool(solve and solve.participant_type in IN_CONTEST),
            first_seen_at=first.creation_time,
            solved_at=solve.creation_time if solve else None,
        ))
    return episodes


def merge(old: Episode | None, new: Episode) -> Episode:
    if old is None:
        return new
    return Episode(
        user_id=old.user_id,
        problem_id=old.problem_id,
        solved=old.solved or new.solved,
        n_attempts=old.n_attempts + new.n_attempts,
        first_verdict=old.first_verdict,                 # earliest overall
        solved_in_contest=old.solved_in_contest if old.solved else new.solved_in_contest,
        first_seen_at=old.first_seen_at,
        solved_at=old.solved_at if old.solved else new.solved_at,
    )
