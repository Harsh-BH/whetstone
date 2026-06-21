"""Load + split interaction records for knowledge-model eval (docs/07 A1)."""

from dataclasses import dataclass


@dataclass
class Record:
    b: float
    tags: list[str]
    y: int
    t: object  # sort key (first_seen_at)


def load_records(conn, user_id: str) -> list["Record"]:
    rows = conn.execute(
        """
        SELECT p.rating AS b, p.tags AS tags, i.first_verdict AS fv, i.first_seen_at AS t
        FROM interactions i JOIN problems p ON p.id = i.problem_id
        WHERE i.user_id = %s AND p.rating IS NOT NULL
          AND p.tags IS NOT NULL AND array_length(p.tags, 1) > 0
          AND i.first_verdict IS NOT NULL
        ORDER BY i.first_seen_at
        """,
        (user_id,),
    ).fetchall()
    return [
        Record(b=float(r["b"]), tags=list(r["tags"]), y=1 if r["fv"] == "OK" else 0, t=r["t"])
        for r in rows
    ]


def temporal_split(recs: list["Record"], frac: float = 0.8) -> tuple[list, list]:
    k = int(len(recs) * frac)
    return recs[:k], recs[k:]
