"""Mastery criterion (docs/02 P5, docs/03): a topic is mastered iff the posterior is
confidently at/above the goal band AND retention is sustained.

  μ_t ≥ R_band  ∧  σ_t ≤ MASTERY_SD_MAX  ∧  stability_t ≥ MASTERY_MIN_STABILITY

The stability floor is the M3 proxy for "retention holds across ≥2 spaced reviews"
(docs/02 P5). `reviews` is the FSRS state dict {tag: {"stability": ...}}.
"""

from config import MASTERY_MIN_STABILITY, MASTERY_SD_MAX


def is_mastered(model, reviews: dict[str, dict], tag: str, r_band: float) -> bool:
    sk = model.tags.get(tag)
    if sk is None:
        return False
    stability = (reviews.get(tag) or {}).get("stability", 0.0)
    return sk.mu >= r_band and sk.sigma <= MASTERY_SD_MAX and stability >= MASTERY_MIN_STABILITY


def mastered_set(model, reviews: dict[str, dict], tags, r_band: float) -> set[str]:
    return {t for t in tags if is_mastered(model, reviews, t, r_band)}


def mark_mastery(conn, user_id: str, model, reviews: dict[str, dict], r_band: float) -> int:
    """Write `mastered` onto the latest topic_skill snapshot. Returns # mastered."""
    mastered = mastered_set(model, reviews, list(model.tags.keys()), r_band)
    conn.execute(
        """
        UPDATE topic_skill SET mastered = (tag = ANY(%s))
        WHERE user_id = %s
          AND snapshot_at = (SELECT max(snapshot_at) FROM topic_skill WHERE user_id = %s)
        """,
        (list(mastered), user_id, user_id),
    )
    return len(mastered)
