"""Orchestrate one ingest run: CF -> normalize/merge -> Postgres (docs/05 step 1).

ponytail: fetches user.status and filters by the stored cursor each run, then
merges new-only episodes with stored ones (correct because submissions are
chronological). Cursor + merge keep re-runs cheap and idempotent.
"""

from datetime import datetime, timezone

from config import settings
from ingest import db, normalize
from ingest.cf_client import CFClient, CFProblem, CFSubmission

CATALOG_MAX_AGE_S = 86400


def _ts(seconds: int | None) -> datetime | None:
    return datetime.fromtimestamp(seconds, tz=timezone.utc) if seconds is not None else None


def _to_db_row(ep: normalize.Episode) -> dict:
    return {
        "user_id": ep.user_id,
        "problem_id": ep.problem_id,
        "solved": ep.solved,
        "n_attempts": ep.n_attempts,
        "first_verdict": ep.first_verdict,
        "solved_in_contest": ep.solved_in_contest,
        "first_seen_at": _ts(ep.first_seen_at),
        "solved_at": _ts(ep.solved_at),
    }


def _problem_row(p: CFProblem, solved_count: int | None) -> dict:
    return {
        "id": p.pid,
        "contest_id": p.contest_id,
        "idx": p.index,
        "name": p.name,
        "rating": p.rating,
        "tags": p.tags,
        "solved_count": solved_count,
    }


def _refresh_catalog(conn, client) -> bool:
    age = db.catalog_age_seconds(conn)
    if age is not None and age <= CATALOG_MAX_AGE_S:
        return False
    problems, solved = client.problemset_problems()
    rows = [_problem_row(p, solved.get(p.pid)) for p in problems if p.contest_id is not None]
    db.upsert_problems(conn, rows)
    return True


def _existing_to_episode(row: dict) -> normalize.Episode:
    def secs(dt):
        return int(dt.timestamp()) if dt else None

    return normalize.Episode(
        user_id=row["user_id"],
        problem_id=row["problem_id"],
        solved=row["solved"],
        n_attempts=row["n_attempts"],
        first_verdict=row["first_verdict"],
        solved_in_contest=row["solved_in_contest"],
        first_seen_at=secs(row["first_seen_at"]),
        solved_at=secs(row["solved_at"]),
    )


def run(handle: str | None = None, client: CFClient | None = None) -> dict:
    handle = handle or settings.cf_handle
    if not handle:
        raise ValueError("CF handle required (set CF_HANDLE or pass handle=)")
    owns_client = client is None
    client = client or CFClient()
    conn = db.connect()
    try:
        refreshed = _refresh_catalog(conn, client)

        cursor = db.get_cursor(conn, handle)
        all_subs: list[CFSubmission] = client.user_status(handle)
        new_subs = [s for s in all_subs if s.creation_time > cursor]

        # FK safety: ensure problems referenced by new submissions exist.
        seen: dict[str, CFProblem] = {}
        for s in new_subs:
            if s.problem.contest_id is not None:
                seen[s.problem.pid] = s.problem
        if seen:
            db.insert_problems_if_absent(conn, [_problem_row(p, None) for p in seen.values()])

        new_eps = normalize.normalize(handle, new_subs)
        new_solves = 0
        for ep in new_eps:
            existing = db.get_interaction(conn, handle, ep.problem_id)
            old = _existing_to_episode(existing) if existing else None
            merged = normalize.merge(old, ep)
            db.upsert_interaction(conn, _to_db_row(merged))
            if merged.solved and not (old and old.solved):
                new_solves += 1

        max_ct = max((s.creation_time for s in new_subs), default=cursor)
        if max_ct > cursor:
            db.set_cursor(conn, handle, max_ct)
        conn.commit()
        return {
            "new_episodes": len(new_eps),
            "new_solves": new_solves,
            "catalog_refreshed": refreshed,
            "cursor": max_ct,
        }
    finally:
        conn.close()
        if owns_client:
            client.close()


if __name__ == "__main__":
    print(run())
