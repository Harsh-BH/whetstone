"""Train mode: maximize expected learning gain per unit time (docs/04 Mode B).

Topic selection: gap_t = max(0, R_band − μ_t), weighted by goal relevance, restricted
to the prereq frontier, then sampled ∝ gap·weight (stochastic so `propensity` is a real
probability for OPE — docs/06). Problem selection: among unseen in-band candidates,
sample weighted by quality (solvedCount).
"""

from recommender.candidates import Candidate


def goal_weights(conn, tags: list[str], r_band: float, band: float = 200.0) -> dict[str, float]:
    """Rating-mode relevance: each tag's frequency in the catalog near the target band,
    normalized over `tags`. (Interview-mode = fixed FAANG weights; config-switchable.)"""
    rows = conn.execute(
        """
        SELECT tag, count(*) AS n
        FROM problems p, unnest(p.tags) AS tag
        WHERE p.rating BETWEEN %(lo)s AND %(hi)s AND tag = ANY(%(tags)s)
        GROUP BY tag
        """,
        {"lo": r_band - band, "hi": r_band + band, "tags": list(tags)},
    ).fetchall()
    counts = {r["tag"]: r["n"] for r in rows}
    total = sum(counts.values()) or 1
    return {t: counts.get(t, 0) / total for t in tags}


def topic_scores(model, open_tags, r_band: float, weights: dict[str, float]) -> dict[str, float]:
    scores = {}
    for t in open_tags:
        sk = model.tags.get(t)
        mu = sk.mu if sk is not None else model.prior_mu
        gap = max(0.0, r_band - mu)
        scores[t] = gap * weights.get(t, 0.0)
    return scores


def _weighted_draw(pool: dict, rng) -> tuple[object, float]:
    """Pick one key ∝ value; return (key, probability it was drawn)."""
    total = sum(pool.values())
    r = rng.random() * total
    acc = 0.0
    for key, w in pool.items():
        acc += w
        if r <= acc:
            return key, w / total
    key = next(reversed(pool))
    return key, pool[key] / total


def sample_topics(scores: dict[str, float], k: int, rng) -> list[tuple[str, float]]:
    """Sample up to k topics WITHOUT replacement ∝ score. Returns (topic, propensity)."""
    pool = {t: s for t, s in scores.items() if s > 0}
    chosen: list[tuple[str, float]] = []
    for _ in range(min(k, len(pool))):
        if not pool:
            break
        t, prob = _weighted_draw(pool, rng)
        chosen.append((t, prob))
        del pool[t]
    return chosen


def pick_in_band(
    model, cands: list[Candidate], rng, band: tuple[float, float]
) -> tuple[Candidate | None, float]:
    """Sample one unseen candidate whose predicted first-attempt P falls in `band`,
    weighted by quality (solvedCount). Returns (candidate, propensity) or (None, 0)."""
    lo, hi = band
    items: list[Candidate] = []
    pool: dict[int, float] = {}  # index -> quality weight (Candidate isn't hashable)
    for c in cands:
        p, _ = model.predict_solve(c.b, c.tags)
        if lo <= p <= hi:
            pool[len(items)] = float(c.solved_count or 1)
            items.append(c)
    if not pool:
        return None, 0.0
    idx, prob = _weighted_draw(pool, rng)
    return items[idx], prob
