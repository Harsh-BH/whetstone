"""Unseen-problem candidate pool from the cached catalog (docs/04)."""

from dataclasses import dataclass


@dataclass
class Candidate:
    pid: str
    b: float
    tags: list[str]
    solved_count: int | None


def load_unseen(
    conn,
    user_id: str,
    tags: list[str] | None = None,
    b_range: tuple[float, float] | None = None,
) -> list[Candidate]:
    """Catalog problems with a rating + tags that the user has NOT interacted with.
    Optionally restrict to problems overlapping `tags` and/or within a `b` window."""
    sql = ["""
        SELECT p.id, p.rating, p.tags, p.solved_count
        FROM problems p
        WHERE p.rating IS NOT NULL AND p.tags IS NOT NULL AND array_length(p.tags, 1) > 0
          AND NOT EXISTS (
              SELECT 1 FROM interactions i WHERE i.user_id = %(uid)s AND i.problem_id = p.id
          )
        """]
    params: dict = {"uid": user_id}
    if tags:
        sql.append("AND p.tags && %(tags)s::text[]")
        params["tags"] = list(tags)
    if b_range:
        sql.append("AND p.rating BETWEEN %(blo)s AND %(bhi)s")
        params["blo"], params["bhi"] = b_range
    rows = conn.execute("\n".join(sql), params).fetchall()
    return [
        Candidate(
            pid=r["id"], b=float(r["rating"]), tags=list(r["tags"]), solved_count=r["solved_count"]
        )
        for r in rows
    ]
