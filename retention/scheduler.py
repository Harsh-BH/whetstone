"""Retention scheduling (docs/02 P3): cold-start FSRS state from solve history, persist
to `reviews`, and surface the due-review queue (concepts whose retrievability has decayed
below the target).
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from config import FSRS_INIT_DIFFICULTY, FSRS_INIT_STABILITY, TARGET_RETRIEVABILITY
from retention import fsrs


def cold_start_state(conn, user_id: str) -> dict[str, dict]:
    """Per tag, replay the user's solved-problem timestamps as successful reviews to
    derive current FSRS state (stability, last_review, due_at)."""
    rows = conn.execute(
        """
        SELECT tag, i.solved_at
        FROM interactions i JOIN problems p ON p.id = i.problem_id, unnest(p.tags) AS tag
        WHERE i.user_id = %s AND i.solved AND i.solved_at IS NOT NULL
        ORDER BY i.solved_at
        """,
        (user_id,),
    ).fetchall()
    by_tag: dict[str, list[datetime]] = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r["solved_at"])

    state: dict[str, dict] = {}
    for tag, times in by_tag.items():
        s = FSRS_INIT_STABILITY
        last = times[0]
        for t in times[1:]:
            elapsed = (t - last).total_seconds() / 86400.0
            s = fsrs.next_stability(s, success=True, elapsed_days=elapsed)
            last = t
        state[tag] = {
            "stability": s,
            "difficulty": FSRS_INIT_DIFFICULTY,
            "last_review": last,
            "due_at": last + timedelta(days=fsrs.days_until_due(s)),
        }
    return state


def save_reviews(conn, user_id: str, state: dict[str, dict]) -> int:
    rows = [
        (user_id, tag, st["stability"], st["difficulty"], st["last_review"], st["due_at"])
        for tag, st in state.items()
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO reviews (user_id, concept, stability, difficulty, last_review, due_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, concept) DO UPDATE SET
                stability=EXCLUDED.stability, difficulty=EXCLUDED.difficulty,
                last_review=EXCLUDED.last_review, due_at=EXCLUDED.due_at
            """,
            rows,
        )
    return len(rows)


def due_queue(conn, user_id: str, now: datetime | None = None) -> list[dict]:
    """Concepts whose current retrievability is below target, most-decayed first."""
    now = now or datetime.now(timezone.utc)
    rows = conn.execute(
        "SELECT concept, stability, last_review, due_at FROM reviews WHERE user_id = %s",
        (user_id,),
    ).fetchall()
    out = []
    for r in rows:
        elapsed = (now - r["last_review"]).total_seconds() / 86400.0
        rr = fsrs.retrievability(float(r["stability"]), elapsed)
        if rr < TARGET_RETRIEVABILITY:
            out.append({"concept": r["concept"], "retrievability": rr, "due_at": r["due_at"]})
    out.sort(key=lambda x: x["retrievability"])
    return out
