"""Compose the daily set: choose modes by uncertainty, blend train/stretch, interleave
topics, and log every served problem for OPE (docs/04 daily-set composition, docs/06).

M2 blend = train + stretch (no FSRS reviews yet; M3 adds the 65/20/15 review slice).
"""

from collections import defaultdict

from config import (
    ASSESS_SIGMA_THRESHOLD,
    DAILY_BLEND,
    FRONTIER_MARGIN,
    MAX_CONSECUTIVE_SAME_TAG,
    MINUTES_PER_PROBLEM,
    STRETCH_TARGET_P,
    TRAIN_TARGET_BAND,
)
from recommender import assess, prereq_dag, train
from recommender.candidates import load_unseen

STRETCH_BAND = (max(0.0, STRETCH_TARGET_P - 0.1), STRETCH_TARGET_P + 0.1)


def daily_set_size(h_hours: float) -> int:
    return max(1, round(h_hours * 60 / MINUTES_PER_PROBLEM / 7))


def _interleave(items: list[dict]) -> list[dict]:
    """Reorder so no two consecutive items share a topic when avoidable (P4)."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        buckets[it["topic"]].append(it)
    out: list[dict] = []
    last = None
    while any(buckets.values()):
        avail = [t for t, v in buckets.items() if v and t != last] or [
            t for t, v in buckets.items() if v
        ]
        topic = max(avail, key=lambda t: len(buckets[t]))
        out.append(buckets[topic].pop(0))
        last = topic
    return out


def _why(mode: str, topic: str, model, r_band: float, p: float) -> str:
    sk = model.tags.get(topic)
    mu = sk.mu if sk else model.prior_mu
    if mode == "assess":
        sd = sk.sigma if sk else model.prior_sigma
        return f"Assess {topic}: pin down skill (μ≈{mu:.0f}±{sd:.0f}); P(solve)≈{p:.0%}."
    gap = max(0.0, r_band - mu)
    kind = "stretch" if p < TRAIN_TARGET_BAND[0] else "train"
    return f"{kind.title()} {topic}: {gap:.0f} below target; P(solve)≈{p:.0%}."


def insert_recommendation(conn, user_id: str, item: dict) -> None:
    conn.execute(
        """
        INSERT INTO recommendations
            (user_id, problem_id, mode, predicted_p, predicted_info, propensity, served_at)
        VALUES (%s, %s, %s, %s, %s, %s, now())
        """,
        (
            user_id,
            item["pid"],
            item["mode"],
            item["predicted_p"],
            item.get("predicted_info"),
            item["propensity"],
        ),
    )


def daily_set(conn, model, user_id: str, r_band: float, h_hours: float, rng) -> list[dict]:
    k = daily_set_size(h_hours)
    all_tags = list(model.tags.keys())
    if not all_tags:
        return []
    open_tags = prereq_dag.frontier(model, all_tags, r_band, FRONTIER_MARGIN)
    if not open_tags:
        return []

    high_sigma = {t for t in open_tags if model.tags[t].sigma > ASSESS_SIGMA_THRESHOLD}
    n_assess = min(len(high_sigma), round(k / 3)) if high_sigma else 0
    if high_sigma and n_assess == 0 and k >= 1:
        n_assess = 1
    n_train_total = k - n_assess
    n_stretch = round(n_train_total * DAILY_BLEND["stretch"])
    n_band = n_train_total - n_stretch

    items: list[dict] = []

    # --- Assess slots ---
    if n_assess:
        cands = load_unseen(conn, user_id, tags=list(high_sigma))
        for d in assess.assess_select(model, cands, high_sigma, n_assess):
            c = d["candidate"]
            topic = next((t for t in c.tags if t in high_sigma), c.tags[0])
            items.append(
                {
                    "pid": c.pid,
                    "b": c.b,
                    "tags": c.tags,
                    "topic": topic,
                    "mode": "assess",
                    "predicted_p": d["predicted_p"],
                    "predicted_info": d["predicted_info"],
                    "propensity": 1.0 / max(1, len(cands)),  # uniform over the assess pool
                    "why": _why("assess", topic, model, r_band, d["predicted_p"]),
                }
            )

    # --- Train + stretch slots (topics being Assessed are not also Trained) ---
    train_tags = open_tags - high_sigma
    weights = train.goal_weights(conn, list(train_tags), r_band)
    scores = train.topic_scores(model, train_tags, r_band, weights)
    sampled = train.sample_topics(scores, n_train_total, rng)
    for i, (topic, topic_prop) in enumerate(sampled):
        band = TRAIN_TARGET_BAND if i < n_band else STRETCH_BAND
        sk = model.tags[topic]
        cands = load_unseen(conn, user_id, tags=[topic], b_range=(sk.mu - 700, sk.mu + 300))
        pick, prob = train.pick_in_band(model, cands, rng, band)
        if pick is None:
            continue
        p, info = model.predict_solve(pick.b, pick.tags)
        items.append(
            {
                "pid": pick.pid,
                "b": pick.b,
                "tags": pick.tags,
                "topic": topic,
                "mode": "train",
                "predicted_p": p,
                "predicted_info": info,
                "propensity": topic_prop * prob,
                "why": _why("train", topic, model, r_band, p),
            }
        )

    items = _interleave(items)
    for it in items:
        insert_recommendation(conn, user_id, it)
    conn.commit()
    return items


def max_consecutive_same_tag(items: list[dict]) -> int:
    """Diagnostic: longest run of consecutive items sharing a topic (should be <= 1)."""
    longest = run = 0
    last = None
    for it in items:
        run = run + 1 if it["topic"] == last else 1
        longest = max(longest, run)
        last = it["topic"]
    return longest


# expose the configured cap for tests/readers
CONFIGURED_MAX_CONSECUTIVE = MAX_CONSECUTIVE_SAME_TAG
