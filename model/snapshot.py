"""Fit the live skill state on all of a user's data, persist a topic_skill snapshot,
and dump radar JSON for the dashboard. This is the `make train` entry point.

Uses the same aggregation+temperature selection as eval (on all data, no split),
so the live model matches what the gate validated.
"""

import json

from eval.dataset import load_records
from eval.run_m1 import _replay_train, select_agg_and_s
from model.irt import SkillModel


def fit_user(conn, user_id: str, cf_rating: float | None) -> SkillModel:
    prior_mu = float(cf_rating) if cf_rating else SkillModel().prior_mu
    recs = load_records(conn, user_id)
    if not recs:
        return SkillModel(prior_mu=prior_mu)
    agg, s = select_agg_and_s(recs, prior_mu)
    return _replay_train(recs, s, prior_mu, agg=agg)


def save_snapshot(conn, user_id: str, model: SkillModel) -> int:
    rows = [(user_id, tag, sk.mu, sk.sigma) for tag, sk in model.tags.items()]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO topic_skill (user_id, tag, mu, sigma, mastered, snapshot_at)
               VALUES (%s, %s, %s, %s, false, now())""",
            rows,
        )
    return len(rows)


def dump_radar(model: SkillModel, path: str) -> None:
    data = [{"tag": t, "mu": sk.mu, "sigma": sk.sigma} for t, sk in sorted(model.tags.items())]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import os

    from ingest import db
    from ingest.cf_client import CFClient

    user = os.environ.get("CF_HANDLE") or "SecondThread"
    client = CFClient()
    try:
        rating = client.user_info(user).rating
    except Exception:
        rating = None
    finally:
        client.close()
    conn = db.connect()
    try:
        m = fit_user(conn, user, rating)
        n = save_snapshot(conn, user, m)
        conn.commit()
    finally:
        conn.close()
    os.makedirs("web/public", exist_ok=True)
    dump_radar(m, "web/public/skills.json")
    print(f"fit {len(m.tags)} tags for {user}; snapshot rows={n}; radar -> web/public/skills.json")
